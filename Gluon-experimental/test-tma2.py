"""
TMA in Gluon
============

The main problem with global memory accesses is register pressure. For each
`LDG.E` or `STG.E`, we need to compute the 64-bit address, compute the mask if
needed, and store the result in registers. Vectorization can reduce register
pressure, but the problem remains.

On Hopper and newer, TMA (Tensor Memory Accelerator) is a hardware feature for
addressing N-dimensional arrays in global memory. TMAs trade the addressing
flexibility of regular global memory instructions for a more concise address
representation -- the "tensor descriptor".

TMAs memory transactions are also handled by a separate hardware path called the
"async proxy". This boosts the performance of global memory accesses, but it
adds an additional layer of synchronization needed.

In this tutorial, we will cover how to use TMAs in Gluon, demonstrate how they
boost performance, and how to pipeline with TMAs.
"""

import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

# Re-use utilities from the previous tutorial.
t3 = importlib.import_module("03-async-copy")


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")






# %%
# Let's rewrite the pipelined elementwise add kernel using TMAs. The structure
# of the kernel is almost the same. However, we now need to allocate one
# mbarrier per buffer to track completion of the reads. We will also use TMA for
# the store, meaning we need to allocate more shared memory for it.
#
# TMAs access shared memory through a different hardware called the "async
# proxy". However, reading and writing shared memory from registers accesses it
# through the "generic proxy". Memory operations across proxies are not ordered,
# so we have to use `fence_async_shared` to establish ordering. Here are some
# examples of hazards that require fences:
#
# ```python
# value = smem.load()
# fence_async_shared()
# tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)
# ```
#
# Without the fence, async_copy_global_to_shared can start copying into `smem`
# while the shared memory load is still in progress.
#
# ```python
# smem.store(value)
# fence_async_shared()
# tma.async_copy_shared_to_global(desc, [0, 0], smem)
# ```
#
# Without the fence, async_copy_shared_to_global can start copying from `smem`
# before the shared memory store is complete.
#
# Note that certain cases imply total completion of a memory transaction and
# do not require a fence. For example, waiting on the result of a TMA load:
#
# ```python
# tma.async_copy_global_to_shared(desc, [0, 0], bar, smem)
# mbarrier.wait(bar, phase=0)
# value = smem.load()
# ```
#
# fence_async_shared is not needed because after the mbarrier.wait on the TMA
# read barrier, we know it has finished writing into shared memory via the async
# proxy. Thus the read via the generic proxy will be ordered after. This applies
# specifically to the TMA read barrier, a fence is still needed in this case:
#
# ```python
# smem.store(value)
# mbarrier.arrive(bar, count=1)
# mbarrier.wait(bar, phase=0)
# fence_async_shared()
# tma.async_copy_shared_to_global(desc, [0, 0], smem)
# ```


@gluon.jit
def issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # Track completion of both TMA reads with the same mbarrier.
    yoff = copy_index * YBLOCK
    bar = bars.index(copy_index % num_buffers)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [xoff, yoff], bar, a_smem.index(copy_index % num_buffers))
    tma.async_copy_global_to_shared(b_desc, [xoff, yoff], bar, b_smem.index(copy_index % num_buffers))
    return copy_index + 1


@gluon.jit
def perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout: gl.constexpr, YBLOCK: gl.constexpr,
                num_buffers: gl.constexpr):
    # Wait for the copy from num_buffers-1 iterations ago to complete.
    read_phase = read_index // num_buffers & 1
    mbarrier.wait(bars.index(read_index % num_buffers), read_phase)
    a_val = a_smem.index(read_index % num_buffers).load(layout)
    b_val = b_smem.index(read_index % num_buffers).load(layout)
    c_val = a_val + b_val
    yoff = read_index * YBLOCK
    # Pipeline the store by rotating the store wait.
    tma.store_wait(pendings=0)
    c_smem.store(c_val)
    fence_async_shared()
    # Issue the store without waiting for it.
    tma.async_copy_shared_to_global(c_desc, [xoff, yoff], c_smem)
    return read_index + 1


@gluon.jit
def elementwise_add_tma_kernel(  #
        a_desc, b_desc, c_desc, xnumel, ynumel,  #
        XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, num_buffers: gl.constexpr):
    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0])
    xoff = pid * XBLOCK

    dtype: gl.constexpr = a_desc.type.block_type.element_ty
    # Allocate multibuffered shared memory for the input buffers.
    a_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buffers, XBLOCK, YBLOCK], b_desc.layout)

    # Allocate shared memory for the TMA store.
    c_smem = gl.allocate_shared_memory(dtype, [XBLOCK, YBLOCK], c_desc.layout)

    # Allocate mbarriers to track completion of the TMA reads.
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)

    copy_index = 0
    read_index = 0

    for _ in gl.static_range(num_buffers - 1):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)

    for _ in range(gl.cdiv(ynumel, YBLOCK) - (num_buffers - 1)):
        copy_index = issue_loads(copy_index, a_desc, b_desc, a_smem, b_smem, bars, xoff, YBLOCK, num_buffers)
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for _ in gl.static_range(num_buffers - 1):
        read_index = perform_add(read_index, bars, a_smem, b_smem, c_smem, c_desc, xoff, layout, YBLOCK, num_buffers)

    for i in gl.static_range(num_buffers):
        mbarrier.invalidate(bars.index(i))

    # Wait for the last store to complete.
    tma.store_wait(pendings=0)


def elementwise_add_tma(a, b, c, XBLOCK=32, YBLOCK=64, num_buffers=2):
    assert a.shape == b.shape == c.shape
    xnumel, ynumel = a.shape
    grid = (triton.cdiv(xnumel, XBLOCK), )

    block_shape = [XBLOCK, YBLOCK]
    # TMA descriptors require NVMMASharedLayout.
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # The strides of TMA descriptors must be 16-byte aligned.
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)
    elementwise_add_tma_kernel[grid](a_desc, b_desc, c_desc, xnumel, ynumel, XBLOCK, YBLOCK, num_buffers)


# %%
# Let's compare the pipelined TMA kernel against the pipelined async copy kernel
# from the previous tutorial.

if __name__ == "__main__":
    print("Benchmarking elementwise_add")
    print("============================")
    xnumel, ynumel = 32 * 1024, 32 * 1024
    A = torch.randn(xnumel, ynumel, device="cuda")
    B = torch.randn(xnumel, ynumel, device="cuda")
    C = torch.empty_like(A, device="cuda")

    XBLOCK = 32
    YBLOCK = 64
    num_buffers = 2

    ms = triton.testing.do_bench(lambda: t3.elementwise_add_pipelined(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_pipelined: {t3.get_throughput(ms, C):.2f} TB/s")

    ms = triton.testing.do_bench(lambda: elementwise_add_tma(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_tma: {t3.get_throughput(ms, C):.2f} TB/s")

# %%
# ```
# elementwise_add_pipelined: 4.20 TB/s
# elementwise_add_tma: 5.50 TB/s
# ```
#
# Switching to TMAs already yields a large performance boost.
#
# Since our kernel has more register room, we can increase the block size. In
# practice, peak register usage will remain low, because the compiler will
# interleave the smem load, add, and smem store in the inner loop. The main
# limitation to block size is the amount of shared memory.
#
# Each SM has 228 KB of shared memory. If we use 128x128xf32 blocks, we don't
# have enough shared memory to double buffer the inputs. If we use 64x128xf32
# triple buffering uses 224 KB, just barely fitting.

if __name__ == "__main__":
    XBLOCK = 64
    YBLOCK = 128
    num_buffers = 3
    ms = triton.testing.do_bench(lambda: elementwise_add_tma(A, B, C, XBLOCK, YBLOCK, num_buffers))
    print(f"elementwise_add_tma (64x128x3): {t3.get_throughput(ms, C):.2f} TB/s")