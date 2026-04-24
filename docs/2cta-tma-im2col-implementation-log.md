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
