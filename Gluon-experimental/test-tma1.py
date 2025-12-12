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

import os
import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

# Re-use utilities from the previous tutorial.
# t3 = importlib.import_module("03-async-copy")


os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_DUMP_DIR'] = 'dump_tma1'
os.environ['TRITON_KERNEL_DUMP'] = '1'

os.environ['MLIR_DUMP_PATH'] = 'dump_tma1_mlir'
os.environ['MLIR_ENABLE_DUMP'] = '1'



def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")





@gluon.jit
def memcpy_1d_tma_kernel(in_desc, out_desc, XBLOCK: gl.constexpr):
    # We don't need to pass the tensor strides because they are stored in the
    # tensor descriptors
    pid = gl.program_id(0)

    # Each tensor descriptor contains a shared memory layout. Data is
    # transferred between global and shared memory according to that layout.
    smem_layout: gl.constexpr = in_desc.layout
    smem = gl.allocate_shared_memory(in_desc.dtype, [XBLOCK], smem_layout)

    # Completion of async TMA reads are tracked by mbarrier objects. These
    # are 64-bit objects that live in shared memory.
    #
    # An mbarrier is initialized with a count. Each time a mbarrier is
    # "arrived" on, the count is decremented. When the count reaches 0, the
    # current phase of the mbarrier is marked as complete and it moves to the
    # next phase. The mbarrier only tracks the state of the current and
    # previous phase. This is important, because if an mbarrier's phase races
    # too far ahead, its waiter will become out of sync.
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    # Completion of an async TMA arrives on an mbarrier once. Thus, initialize
    # the mbarrier with a count of 1 so its phase will complete when the TMA is
    # complete.
    mbarrier.init(bar, count=1)

    # Tensor descriptors have an associated block shape. Each TMA request will
    # copy one block of the tensor descriptor. The coordinates of the TMA
    # request are specified as offsets to the beginning of the block. Masking
    # of out-of-bounds reads and writes is handled automatically by TMAs, using
    # the shape specified on the tensor descriptor.
    gl.static_assert(in_desc.block_type == out_desc.block_type)
    gl.static_assert(in_desc.layout == out_desc.layout)

    # Track completion of the TMA read based on the number of bytes copied.
    # mbarrier.expect sets the number of outstanding bytes tracked by the
    # mbarrier. If we pass the barrier to the TMA copy, it will atomically
    # decrement the number of outstanding bytes as transactions complete. When
    # it reaches 0, the mbarrier is arrived on once.
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(in_desc, [pid * XBLOCK], bar, smem)

    # Wait for completion of the read. We query the completion state of the
    # mbarrier using the parity of the phase, i.e. either 0 or 1. mbarriers are
    # initialized to parity 1 complete, so we wait for parity 0.
    mbarrier.wait(bar, phase=0)

    # When we are done using the mbarrier, we need to invalidate it.
    mbarrier.invalidate(bar)

    # Since the TMA store reads from shared memory, we don't even need to load
    # the result into registers. We can just store the result directly.
    tma.async_copy_shared_to_global(out_desc, [pid * XBLOCK], smem)

    # Unlike TMA reads, the completion of TMA stores is tracked by commit
    # groups, just like async copies. Each async TMA store is implicitly
    # committed to an async store group. We can wait until there are at most
    # `pendings` outstanding TMA stores using `store_wait`. Note that the commit
    # groups for async copy and async TMA stores are separate.
    tma.store_wait(pendings=0)



def memcpy_1d_tma(input, output, XBLOCK=8192):
    assert input.shape == output.shape

    # The layout for a tensor descriptor is always an NVMMASharedLayout. We can
    # use this helper to grab the default NVMMASharedLayout, but sometimes you
    # might need a different layout.
    block_shape = [XBLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # Wrap the tensors in tensor descriptors.
    in_desc = TensorDescriptor.from_tensor(input, block_shape, layout)
    out_desc = TensorDescriptor.from_tensor(output, block_shape, layout)

    grid = (triton.cdiv(input.numel(), XBLOCK), )
    # Our kernel only uses scalars, so just a single warp is enough.
    memcpy_1d_tma_kernel[grid](in_desc, out_desc, XBLOCK, num_warps=1)



def test_memcpy_1d_tma(XBLOCK, xnumel):
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)
    memcpy_1d_tma(input, output, XBLOCK)
    torch.testing.assert_close(input, output, atol=0, rtol=0)


if __name__ == "__main__":
    test_memcpy_1d_tma(XBLOCK=64, xnumel=40)