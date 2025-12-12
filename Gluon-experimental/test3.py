import os
import pytest
import torch
import triton
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl



os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_DUMP_DIR'] = 'dump_test2'
os.environ['TRITON_KERNEL_DUMP'] = '1'


def _enabled(label):
    from sys import argv
    return len(argv) == 1 or label in argv[1].split(",")



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



def get_layout_for_gmem_access(tensor, num_warps):
    if len(tensor.shape) == 1:
        return gl.BlockedLayout([1], [32], [num_warps], [0])

    assert len(tensor.shape) == 2, "only 1D and 2D tensors are supported"
    assert 1 in tensor.stride(), "expected at least 1 contiguous dimension"
    if tensor.stride(1) == 1:
        return gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])
    else:
        return gl.BlockedLayout([1, 1], [32, 1], [num_warps, 1], [0, 1])




@gluon.jit
def get_mask_and_offsets(start_x, start_y, xnumel, ynumel, xstride, ystride,  #
                         XBLOCK: gl.constexpr, YBLOCK: gl.constexpr, layout: gl.constexpr):
    indices_x = start_x + gl.arange(0, XBLOCK, layout=gl.SliceLayout(dim=1, parent=layout))
    indices_y = start_y + gl.arange(0, YBLOCK, layout=gl.SliceLayout(dim=0, parent=layout))

    mask = (indices_x[:, None] < xnumel) & (indices_y[None, :] < ynumel)
    offsets = xstride * indices_x[:, None] + ystride * indices_y[None, :]
    return mask, offsets


