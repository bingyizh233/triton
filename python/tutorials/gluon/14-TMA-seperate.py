"""
Per-CTA TMA loads into a 2-CTA MMA tile
=======================================

This mirrors :file:`14-multicta-simple.py`, but loads **A** and **B** with
**1-CTA** tensor descriptors (no ``cga_layout`` on the descriptor). The
cluster shared-memory allocations are then **reinterpreted** as local TMA
memdescs whose ``cga_layout`` is broadcasted (``[(0, 0)]``), while the
original cluster memdescs are kept for the 2-CTA MMA.

Unlike a logical ``smem_a_cta0`` / ``smem_a_cta1`` view, this avoids
``memdesc.slice`` across CTA-sharded dimensions, which is not supported today.
The broadcasted ``cga_layout`` keeps the memdesc valid in a ``num_ctas=2``
kernel while preserving the same per-CTA physical allocation size.

This uses:

- ``cluster.cluster_cta_id()`` to tell CTAs apart (``program_id`` is the
  **cluster** id when ``num_ctas > 1``, not the CTA index);
- per-CTA source coordinates derived from ``cluster.cluster_cta_id()``.

The MMA path is unchanged: operands remain full cluster tiles with
``cga_layout``, and ``tcgen05_mma`` still uses ``two_ctas=True``.
"""

import pytest
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_mma,
)
from triton.experimental.gluon.language.nvidia.hopper import cluster, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_blackwell():
    raise RuntimeError("This tutorial requires a Blackwell NVIDIA GPU")


@gluon.jit
def two_cta_tcgen05_separate_tma_kernel(a_desc_1cta, b_desc_1cta, c_desc):
    gl.static_assert(gl.num_ctas() == 2)

    cta_m: gl.constexpr = a_desc_1cta.block_shape[0]
    cluster_m: gl.constexpr = cta_m * 2
    K: gl.constexpr = a_desc_1cta.block_shape[1]
    cta_n: gl.constexpr = b_desc_1cta.block_shape[1]
    tile_n: gl.constexpr = cta_n * 2
    cga_layout: gl.constexpr = c_desc.layout.cga_layout

    a_cluster_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [cluster_m, K], a_desc_1cta.dtype, cga_layout=[(1, 0)]
    )
    b_cluster_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [K, tile_n], b_desc_1cta.dtype, cga_layout=[(0, 1)]
    )
    a_tma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [cta_m, K], a_desc_1cta.dtype, cga_layout=[(0, 0)]
    )
    b_tma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [K, cta_n], b_desc_1cta.dtype, cga_layout=[(0, 0)]
    )

    smem_a = gl.allocate_shared_memory(a_desc_1cta.dtype, [cluster_m, K], a_cluster_layout)
    smem_b = gl.allocate_shared_memory(b_desc_1cta.dtype, [K, tile_n], b_cluster_layout)

    # Reinterpret the cluster allocations as local TMA views. The broadcasted
    # cga_layout keeps the memdesc valid in a 2-CTA kernel.
    smem_a_local = smem_a._reinterpret(a_desc_1cta.dtype, [cta_m, K], a_tma_layout)
    smem_b_local = smem_b._reinterpret(b_desc_1cta.dtype, [K, cta_n], b_tma_layout)

    cid = cluster.cluster_cta_id()
    # The per-CTA coordinate selects which global tile this CTA loads into its
    # local reinterpret view.
    a_coord = [cid * cta_m, 0]
    b_coord = [0, cid * cta_n]

    tma_bar = mbarrier.allocate_mbarrier(two_ctas=True)
    mma_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(tma_bar, count=1)
    mbarrier.init(mma_bar, count=1)

    mbarrier.expect(tma_bar, a_desc_1cta.nbytes_per_cta + b_desc_1cta.nbytes_per_cta)

    tma.async_copy_global_to_shared(a_desc_1cta, a_coord, tma_bar, smem_a_local)
    tma.async_copy_global_to_shared(b_desc_1cta, b_coord, tma_bar, smem_b_local)

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


def run_two_cta_separate_tma(a, b, c):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    assert M % 2 == 0
    cta_m = M // 2
    cta_n = N // 2

    a_layout_1cta = gl.NVMMASharedLayout.get_default_for([cta_m, K], gl.float16)
    b_layout_1cta = gl.NVMMASharedLayout.get_default_for([K, cta_n], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([M, N], gl.float16, cga_layout=[(1, 0)])

    a_desc_1cta = TensorDescriptor.from_tensor(a, [cta_m, K], a_layout_1cta)
    b_desc_1cta = TensorDescriptor.from_tensor(b, [K, cta_n], b_layout_1cta)
    c_desc = TensorDescriptor.from_tensor(c, [M, N], c_layout)

    two_cta_tcgen05_separate_tma_kernel[(1,)](a_desc_1cta, b_desc_1cta, c_desc, num_warps=4, num_ctas=2)


def test_two_cta_separate_tma():
    M, N, K = 256, 128, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    run_two_cta_separate_tma(a, b, c)
    torch.testing.assert_close(c, torch.matmul(a, b), atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    test_two_cta_separate_tma()
