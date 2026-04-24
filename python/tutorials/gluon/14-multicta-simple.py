"""
Multi-CTA
=========

In Hopper, NVIDIA added a new thread group level to the hierarchy, the CGA.
A CGA is a group of up to 16 CTAs that may collaborate with each other.
In particular they can:

- Load data from HBM collaboratively via TMA broadcasting
- Exchange data by accessing each other's shared memory. This is often called
  "using distributed shared memory"
- Starting in Blackwell, pairs CTA can collaboratively compute the result of a
  matrix multiplication
- Subsets of the CGA cluster can be selectively synchronized via `mbarrier`s

Of course, different CTAs may or may not be allocated to the same SM (in fact
the documentation does not provide any guarantees about this) so operations like
synchronisation or accessing each other's shared memory are **much** more costly
than accessing shared memory or synchronising the threads within a single CTA.
As such, when using CGAs, the name of the game is to maximise the collaboration
while not introducing unnecessary synchronisation points.

Multi-CTA Layouts
-----------------

Layouts can be sharded across CTAs in a natural way. For example, we can have a
blocked layout on a program with 4 warps and 2 CTAs of the form:

```
gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0], cga_layout=[[1, 0]])
```

The `cga_layout` representation `[[1, 0]]` denotes a linear layout. In this case, it
denotes that the two CTAs are sharding the tensor along the 0th dimension into two
contiguous subtensors.

Similarly, if we had 8 CTAs and we wanted to shard a shared memory descriptor across
the 0th dimension using the first 4 CTAs and then across the 1st dimension using the
last 4 CTAs, the layout could look like

```python
gl.NVMMASharedLayout.get_default_for([M, N], gl.float16, cga_layout=[[1, 0], [2, 0], [0, 1]])
```

The `cga_layout` will always have `log2(numCTAs)` bases, and it will always denote sharding
the full tensor into contiguous chunks. For more sharding patterns where the CTAs may not
shard the tensor into contiguous subtensors, like

| CTA0 warp0 |
| CTA1 warp0 |
| CTA0 warp1 |
| CTA1 warp1 |

one may use the layouts `LinearEncoding` for data in registers and `SharedLinearEncoding`
for data in shared memory. In these cases, rather than having an attribute called `CGALayout`
the CGA layout is encoded as part of the `LinearLayout` under the input dimension named `block`.
The example above would then look like:

```python
gl.LinearEncoding(warps=[[2]], block=[[1]])
```

as we shard first along the CTAs and then along the warps.
"""

import importlib
import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    clc,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tensor_memory_descriptor,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

# Re-use baseline tutorials for comparisons.
t8 = importlib.import_module("08-warp-specialization")


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")


def tflops(ms, M, N, K):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def gbps(ms, num_bytes):
    return num_bytes * 1e-9 / (ms * 1e-3)


def pick_multicta_softmax_config(n_cols):
    warp_thresholds = [(3072, 1), (6144, 2)]
    cluster_thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]

    num_warps = next((v for limit, v in warp_thresholds if n_cols <= limit), 4)
    cluster_n = next((v for limit, v in cluster_thresholds if n_cols <= limit), 16)
    return {
        "num_warps": num_warps,
        "num_ctas": cluster_n,
    }

@gluon.jit
def two_cta_tcgen05_kernel(a_desc, b_desc, c_desc):
    gl.static_assert(gl.num_ctas() == 2)

    cluster_m: gl.constexpr = a_desc.block_shape[0]
    tile_n: gl.constexpr = b_desc.block_shape[1]
    cta_m: gl.constexpr = cluster_m // 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout

    smem_a = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    smem_b = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    mbarrier.expect(tma_bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
    tma.async_copy_global_to_shared(a_desc, [0, 0], tma_bar, smem_a)
    tma.async_copy_global_to_shared(b_desc, [0, 0], tma_bar, smem_b)
    mbarrier.wait(tma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(tma_bar)

    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(cta_m, tile_n),
        col_stride=1,
        cga_layout=cga_layout,
        two_ctas=True,
    )
    acc = allocate_tensor_memory(gl.float32, [cluster_m, tile_n], acc_layout)

    tcgen05_mma(smem_a, smem_b, acc, use_acc=False, mbarriers=[mma_bar])
    mbarrier.wait(mma_bar, phase=0, deps=[smem_a, smem_b])
    mbarrier.invalidate(mma_bar)

    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)
    c_smem.store(acc.load().to(c_desc.dtype))
    tma.async_copy_shared_to_global(c_desc, [0, 0], c_smem)


def run_two_cta_tcgen05(a, b, c):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    a_layout = gl.NVMMASharedLayout.get_default_for([M, K], gl.float16, cga_layout=[(1, 0)])
    b_layout = gl.NVMMASharedLayout.get_default_for([K, N], gl.float16, cga_layout=[(0, 1)])
    c_layout = gl.NVMMASharedLayout.get_default_for([M, N], gl.float16, cga_layout=[(1, 0)])

    a_desc = TensorDescriptor.from_tensor(a, [M, K], a_layout)
    b_desc = TensorDescriptor.from_tensor(b, [K, N], b_layout)
    c_desc = TensorDescriptor.from_tensor(c, [M, N], c_layout)

    two_cta_tcgen05_kernel[(1, )](a_desc, b_desc, c_desc, num_warps=4, num_ctas=2)



def test_two_cta_tcgen05():
    M, N, K = 256, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    run_two_cta_tcgen05(a, b, c)
    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    test_two_cta_tcgen05()