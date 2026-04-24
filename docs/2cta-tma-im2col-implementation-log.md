# 2-CTA TMA Im2col Implementation Log

Branch: codex/2cta-im2col-layout-lowering
Base: c6a4eecfc
Started: 2026-04-24T11:47:59-07:00

## Step 0: Branch setup

- Created implementation branch from current 2cta-conv-clc HEAD.
- Preserving existing untracked experiment artifacts; commits will stage only intentional files.

## Step 1: First-class CTA-local shared-memory view

- Added `shared_memory_descriptor.local_cta_view(dtype, shape, layout)` as the public API for viewing a cluster shared-memory allocation as a CTA-local tile.
- Updated `python/examples/gluon/conv-2cta-fprop.py` to use `local_cta_view` instead of calling `_reinterpret` directly in the TMA producer.
- This is intentionally a semantic wrapper over the existing memdesc reinterpret lowering for now; the next steps can add verifier/lowering constraints behind the public API without keeping `_reinterpret` in the convolution code.

## Step 2: Hide 2-CTA im2col M-offset calculation behind TMA API

- Added `tma.async_load_im2col_m_split` / `tma.async_copy_global_to_shared_im2col_m_split`.
- Added `tma.async_load_cta_split` / `tma.async_copy_global_to_shared_cta_split` for ordinary tiled TMA dimensions split across CTAs.
- The new helper accepts cluster-level logical NHWC output coordinates `[batch, out_y, out_x, channel]` plus output shape, stride, and padding.
- It derives the local CTA logical-M offset internally and converts the adjusted logical M position back into `(batch, out_y, out_x)` before issuing the existing im2col TMA op.
- Updated the 2-CTA convolution prototype so user kernel code no longer calls `cluster.cluster_cta_id()` or stores `cid` in `V4Args`; A's M split and B's N split are now hidden behind TMA helpers.
- Verification: `py_compile` passed for the modified TMA frontends and convolution prototype, and both Hopper/Blackwell TMA modules expose the new helpers.
- B200 correctness smoke test was submitted as Slurm job `1958890` but remained pending on resources and was cancelled before allocation.
