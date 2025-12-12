import os
import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl



os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_DUMP_DIR'] = 'dump_test1'
os.environ['TRITON_KERNEL_DUMP'] = '1'


def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")



@gluon.jit
def memcpy_1d_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK

    # The main difference between writing this kernel in Triton and Gluon is
    # we need to specify the layout of the 1D tensor. Layouts are propagated
    # forwards through type inference, so we only need to specify the layout for
    # the indices tensor.
    indices = gl.arange(0, XBLOCK, layout=layout)

    offsets = start + indices
    in_ptrs = in_ptr + offsets
    mask = offsets < xnumel

    value = gl.load(in_ptrs, mask=mask)
    out_ptrs = out_ptr + offsets
    gl.store(out_ptrs, value, mask=mask)


def memcpy_1d_impl(input, output, XBLOCK, layout, num_warps):
    xnumel = input.numel()
    grid = (triton.cdiv(xnumel, XBLOCK), )
    compiled_kernel = memcpy_1d_kernel[grid](input, output, xnumel, XBLOCK, layout, num_warps=num_warps)
    return compiled_kernel




def get_throughput(input, ms):
    tbytes = (2 * input.numel() * input.element_size() >> 30) / 1024
    return tbytes / (ms * 1e-3)


def bench_memcpy_impl(input, output, impl):
    compiled_kernel = impl(input, output)
    fn = lambda: impl(input, output)
    ms = triton.testing.do_bench(fn)
    return compiled_kernel, get_throughput(input, ms)


def bench_memcpy(impl):
    torch.manual_seed(0)
    xnumel = 2 << 30
    input = torch.randn(xnumel, device="cuda")
    output = torch.empty_like(input)

    return bench_memcpy_impl(input, output, impl)




if __name__ == "__main__" and _enabled("R_vs_throughput"):
    print("R vs. Throughput")
    print("================")
    XBLOCK = 2048
    num_warps = 4
    kernel = partial(memcpy_1d_impl, XBLOCK=XBLOCK, num_warps=num_warps)
    compiled_kernels = []
    for i in range(0, 5):
        R = 2**i
        layout = gl.BlockedLayout([R], [32], [num_warps], [0])
        impl = partial(kernel, layout=layout)
        compiled_kernel, throughput = bench_memcpy(impl)
        compiled_kernels.append((R, compiled_kernel))
        print(f"R={R:<3} {throughput:.3f} TB/s")
    print()
