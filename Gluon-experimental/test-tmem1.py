"""
The 5th Generation TensorCore^TM
================================

This tutorial covers the APIs for interacting with Tensor Cores on Blackwell
GPUs. Blackwell Tensor Cores introduce a new memory space called Tensor Memory
that must be used to interact with the async MMA instructions.

In this tutorial, we will cover allocating and interacting with Tensor Memory
and demonstrate how to use the `tcgen05` MMA instructions. We will build a
simple matmul kernel to demonstrate practical uses of the APIs and show an
example of how to pipeline MMA instructions.
"""

import os




import itertools
import pytest
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tma,
    mbarrier,
    tcgen05_mma,
    tcgen05_commit,
    fence_async_shared,
)



os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_DUMP_DIR'] = 'dump_tmem_test1'
os.environ['TRITON_KERNEL_DUMP'] = '1'




def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")




@gluon.jit
def tmem_example_kernel(in_ptr, out_ptr, M: gl.constexpr, N: gl.constexpr, num_warps: gl.constexpr):
    global_memory_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    offs_m = gl.arange(0, M, gl.SliceLayout(1, global_memory_layout))
    offs_n = gl.arange(0, N, gl.SliceLayout(0, global_memory_layout))
    offs = offs_m[:, None] * N + offs_n[None, :]

    input = gl.load(in_ptr + offs)

    # Allocate some tensor memory.
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(64, 64),
        col_stride=32 // in_ptr.dtype.element_ty.primitive_bitwidth,
    )

    tmem = allocate_tensor_memory(
        element_ty=in_ptr.dtype.element_ty,
        shape=[M, N],
        layout=tmem_layout,
    )

    # Get the register layout needed to access the tensor memory using a helper.
    tmem_reg_layout: gl.constexpr = get_tmem_reg_layout(
        in_ptr.dtype.element_ty,
        (M, N),
        tmem_layout,
        num_warps=num_warps,
    )

    input = gl.convert_layout(input, tmem_reg_layout)
    tmem.store(input)
    output = tmem.load(tmem_reg_layout)
    output = gl.convert_layout(output, global_memory_layout)

    gl.store(out_ptr + offs, output)


def test_tmem_example_kernel(M, N, num_warps):
    M = 64
    N = 64
    num_warps = 4
    input = torch.randn(M, N, dtype=torch.float32, device="cuda")
    output = torch.empty_like(input)

    tmem_example_kernel[(1, )](input, output, M, N, num_warps=num_warps)
    torch.testing.assert_close(input, output, atol=0, rtol=0)
