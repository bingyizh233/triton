"""
Implicit GEMM Conv2D in Gluon (tcgen05, pipelined)
==================================================

This kernel mirrors the optimized Triton implementation found in
`implicit_gemm_conv2d.py` and its TTIR dump. It uses the same tiling strategy,
multi-stage async copies, tensor-memory accumulators, and tcgen05 MMA
instructions available on Blackwell GPUs.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
    fence_async_shared,
    mbarrier,
)


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


# -----------------------------------------------------------------------------
# GLUON KERNEL
# -----------------------------------------------------------------------------


@gluon.jit
def _issue_async_tile(
    copy_idx,
    input_ptr,
    weight_ptr,
    a_bufs,
    b_bufs,
    batch_id,
    batch_mask,
    out_y,
    out_x,
    offs_n,
    k_cols,
    k_rows,
    col_layout: gl.constexpr,
    row_layout: gl.constexpr,
    m_layout: gl.constexpr,
    stride_input_n,
    stride_input_h,
    stride_input_w,
    stride_weight_k,
    stride_weight_r,
    stride_weight_s,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    H,
    W,
    K,
    C,
    R,
    S,
    K_GEMM,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    stage = copy_idx % NUM_BUFFERS
    k_start = copy_idx * BLOCK_K
    offs_k_cols = gl.convert_layout(k_start + k_cols, col_layout)
    offs_k_rows = gl.convert_layout(k_start + k_rows, row_layout)
    mask_k = gl.convert_layout(offs_k_cols < K_GEMM, col_layout)

    iter_col_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=iter_layout)
    iter_row_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=iter_layout)

    c_in_cols = gl.convert_layout(offs_k_cols % C, iter_col_layout)
    rem_rs_cols = gl.convert_layout(offs_k_cols // C, iter_col_layout)
    s_filter_cols = gl.convert_layout(rem_rs_cols % S, iter_col_layout)
    r_filter_cols = gl.convert_layout(rem_rs_cols // S, iter_col_layout)

    c_in_rows = gl.convert_layout(offs_k_rows % C, iter_row_layout)
    rem_rs_rows = gl.convert_layout(offs_k_rows // C, iter_row_layout)
    s_filter_rows = gl.convert_layout(rem_rs_rows % S, iter_row_layout)
    r_filter_rows = gl.convert_layout(rem_rs_rows // S, iter_row_layout)

    curr_h = out_y[:, None] * stride_h + r_filter_cols[None, :] - pad_h
    curr_w = out_x[:, None] * stride_w + s_filter_cols[None, :] - pad_w

    mask_h = gl.convert_layout((curr_h >= 0) & (curr_h < H), iter_col_layout)
    mask_w = gl.convert_layout((curr_w >= 0) & (curr_w < W), iter_col_layout)
    mask_batch = gl.convert_layout(batch_mask, iter_col_layout)
    mask_k = gl.convert_layout(mask_k, iter_col_layout)
    load_mask_a = mask_h & mask_w & mask_batch & mask_k[None, :]

    batch_id = gl.convert_layout(batch_id, col_layout)
    input_offsets = (
        batch_id[:, None] * stride_input_n
        + curr_h * stride_input_h
        + curr_w * stride_input_w
        + c_in_cols[None, :]
    )
    cp.async_copy_global_to_shared(
        a_bufs.index(stage), input_ptr + input_offsets, mask=load_mask_a
    )

    offs_n = gl.convert_layout(offs_n, iter_row_layout)
    weight_offsets = (
        offs_n[None, :] * stride_weight_k
        + r_filter_rows[:, None] * stride_weight_r
        + s_filter_rows[:, None] * stride_weight_s
        + c_in_rows[:, None]
    )
    k_mask_rows = gl.convert_layout(mask_k, iter_row_layout)
    weight_mask = (offs_n[None, :] < K) & k_mask_rows[:, None]
    cp.async_copy_global_to_shared(
        b_bufs.index(stage), weight_ptr + weight_offsets, mask=weight_mask
    )

    cp.commit_group()
    return copy_idx + 1


@gluon.jit
def _consume_tile(
    read_idx,
    a_bufs,
    b_bufs,
    acc_tmem,
    mma_bar,
    phase,
    use_acc,
    NUM_BUFFERS: gl.constexpr,
):
    stage = read_idx % NUM_BUFFERS
    fence_async_shared()
    tcgen05_mma(a_bufs.index(stage), b_bufs.index(stage), acc_tmem, use_acc=use_acc)
    tcgen05_commit(mma_bar)
    mbarrier.wait(mma_bar, phase=phase)
    return read_idx + 1, True, phase ^ 1


@gluon.jit
def implicit_gemm_conv2d_gluon_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    H,
    W,
    C,
    K,
    R,
    S,
    out_h,
    out_w,
    stride_input_n,
    stride_input_h,
    stride_input_w,
    stride_weight_k,
    stride_weight_r,
    stride_weight_s,
    stride_output_n,
    stride_output_h,
    stride_output_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    num_warps: gl.constexpr,
):
    gl.static_assert(NUM_BUFFERS >= 2)
    gl.static_assert(num_warps == 8)

    pid = gl.program_id(axis=0)
    M_GEMM = N * out_h * out_w
    N_GEMM = K
    K_GEMM = R * S * C
    if M_GEMM <= 0 or N_GEMM <= 0 or K_GEMM <= 0:
        return

    num_pid_m = gl.cdiv(M_GEMM, BLOCK_M)
    num_pid_n = gl.cdiv(N_GEMM, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    if num_pid_in_group <= 0:
        return

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    if group_size_m <= 0:
        return

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    iter_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [1, 32], [num_warps, 1], [1, 0]
    )
    slice_n: gl.constexpr = gl.SliceLayout(dim=0, parent=iter_layout)
    slice_m: gl.constexpr = gl.SliceLayout(dim=1, parent=iter_layout)

    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=slice_m)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=slice_n)
    k_parent_layout: gl.constexpr = gl.BlockedLayout(
        [1], [BLOCK_K], [1], [0]
    )
    col_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=k_parent_layout)
    row_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=k_parent_layout)
    k_cols = gl.arange(0, BLOCK_K, layout=col_layout)
    k_rows = gl.arange(0, BLOCK_K, layout=row_layout)

    pixels_per_batch = out_h * out_w
    batch_id = offs_m // pixels_per_batch
    m_residual = offs_m % pixels_per_batch
    out_y = m_residual // out_w
    out_x = m_residual % out_w
    out_y = gl.convert_layout(out_y, iter_col_layout)
    out_x = gl.convert_layout(out_x, iter_col_layout)
    batch_mask = gl.convert_layout(batch_id[:, None] < N, iter_col_layout)

    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K], gl.float16
    )
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, BLOCK_N], gl.float16
    )

    a_bufs = gl.allocate_shared_memory(
        gl.float16, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layout=a_smem_layout
    )
    b_bufs = gl.allocate_shared_memory(
        gl.float16, [NUM_BUFFERS, BLOCK_K, BLOCK_N], layout=b_smem_layout
    )

    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(BLOCK_M // 2, BLOCK_N), col_stride=1
    )
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)
    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (BLOCK_M, BLOCK_N), tmem_layout, num_warps
    )

    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    phase = 0

    num_tiles = gl.cdiv(K_GEMM, BLOCK_K)
    if num_tiles <= 0:
        return

    copy_idx = 0
    read_idx = 0
    use_acc = False

    for _ in gl.static_range(NUM_BUFFERS - 1):
        copy_idx = _issue_async_tile(
            copy_idx,
            input_ptr,
            weight_ptr,
            a_bufs,
            b_bufs,
            batch_id,
            batch_mask,
            out_y,
            out_x,
            offs_n,
            k_cols,
            k_rows,
            m_layout,
            iter_col_layout,
            iter_row_layout,
            stride_input_n,
            stride_input_h,
            stride_input_w,
            stride_weight_k,
            stride_weight_r,
            stride_weight_s,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            H,
            W,
            K,
            C,
            R,
            S,
            K_GEMM,
            BLOCK_K,
            NUM_BUFFERS,
        )

    for _ in range(num_tiles - (NUM_BUFFERS - 1)):
        copy_idx = _issue_async_tile(
            copy_idx,
            input_ptr,
            weight_ptr,
            a_bufs,
            b_bufs,
            batch_id,
            batch_mask,
            out_y,
            out_x,
            offs_n,
            k_cols,
            k_rows,
            m_layout,
            iter_col_layout,
            iter_row_layout,
            stride_input_n,
            stride_input_h,
            stride_input_w,
            stride_weight_k,
            stride_weight_r,
            stride_weight_s,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            H,
            W,
            K,
            C,
            R,
            S,
            K_GEMM,
            BLOCK_K,
            NUM_BUFFERS,
        )
        cp.wait_group(NUM_BUFFERS - 1)
        read_idx, use_acc, phase = _consume_tile(
            read_idx, a_bufs, b_bufs, acc_tmem, mma_bar, phase, use_acc, NUM_BUFFERS
        )

    for i in gl.static_range(NUM_BUFFERS - 1):
        cp.wait_group(NUM_BUFFERS - 2 - i)
        if read_idx < num_tiles:
            read_idx, use_acc, phase = _consume_tile(
                read_idx,
                a_bufs,
                b_bufs,
                acc_tmem,
                mma_bar,
                phase,
                use_acc,
                NUM_BUFFERS,
            )

    mbarrier.invalidate(mma_bar)

    acc = acc_tmem.load(acc_reg_layout)
    acc_fp16 = acc.to(gl.float16)

    c_batch = offs_m // pixels_per_batch
    c_residual = offs_m % pixels_per_batch
    c_out_y = c_residual // out_w
    c_out_x = c_residual % out_w

    c_offsets = (
        c_batch[:, None] * stride_output_n
        + c_out_y[:, None] * stride_output_h
        + c_out_x[:, None] * stride_output_w
        + offs_n[None, :]
    )
    c_mask = (offs_m[:, None] < M_GEMM) & (offs_n[None, :] < N_GEMM)
    gl.store(output_ptr + c_offsets, acc_fp16, mask=c_mask)


# -----------------------------------------------------------------------------
# HOST WRAPPER
# -----------------------------------------------------------------------------


def implicit_gemm_conv2d_gluon(input_tensor, weight_tensor, stride=1, padding=0):
    if not is_blackwell():
        raise RuntimeError("This kernel requires a Blackwell-class GPU (SM 10.x)")

    N, H, W, C = input_tensor.shape
    K, R, S, C_in = weight_tensor.shape
    assert C == C_in, "Input and Weight channels must match"

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1

    output = torch.empty((N, out_h, out_w, K), device=input_tensor.device, dtype=torch.float16)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    GROUP_SIZE_M = 4
    NUM_BUFFERS = 3
    NUM_WARPS = 8

    grid = (
        triton.cdiv(N * out_h * out_w, BLOCK_M)
        * triton.cdiv(K, BLOCK_N),
    )

    implicit_gemm_conv2d_gluon_kernel[grid](
        input_tensor,
        weight_tensor,
        output,
        N,
        H,
        W,
        C,
        K,
        R,
        S,
        out_h,
        out_w,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        weight_tensor.stride(0),
        weight_tensor.stride(1),
        weight_tensor.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        stride,
        stride,
        padding,
        padding,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_BUFFERS=NUM_BUFFERS,
        num_warps=NUM_WARPS,
    )

    return output


# -----------------------------------------------------------------------------
# VERIFICATION & BENCHMARK
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    if not is_blackwell():
        print("Please run on a Blackwell GPU (SM 10.x).")
        raise SystemExit(0)

    torch.manual_seed(0)

    N, H, W, C = 128, 64, 64, 384
    K, R, S = 384, 3, 3
    stride = 1
    padding = 1

    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1

    print(
        f"Parameters: N={N}, H={H}, W={W}, C={C}, K={K}, R={R}, S={S}, stride={stride}, padding={padding}"
    )
    print(f"Device: {torch.cuda.get_device_name()}")

    x_nchw = torch.randn((N, C, H, W), device="cuda", dtype=torch.float16)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    w_nchw = torch.randn((K, C, R, S), device="cuda", dtype=torch.float16)
    w_krsc = w_nchw.permute(0, 2, 3, 1).contiguous()

    print("\nRunning Verification...")
    torch.cuda.nvtx.range_push("Gluon tcgen05")
    gluon_out = implicit_gemm_conv2d_gluon(x_nhwc, w_krsc, stride=stride, padding=padding)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("PyTorch Reference")
    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out = torch_out.permute(0, 2, 3, 1)
    torch.cuda.nvtx.range_pop()

    if torch.allclose(gluon_out, torch_out, atol=1e-2, rtol=1e-2):
        print("✅ Match!")
    else:
        diff = (gluon_out - torch_out).abs().max()
        print(f"❌ Mismatch (max diff {diff})")

    print("\nRunning Benchmark...")
    torch.cuda.synchronize()
    for _ in range(5):
        implicit_gemm_conv2d_gluon(x_nhwc, w_krsc, stride=stride, padding=padding)
    torch.cuda.synchronize()

    ms_gluon = triton.testing.do_bench(
        lambda: implicit_gemm_conv2d_gluon(x_nhwc, w_krsc, stride=stride, padding=padding)
    )
    ms_torch = triton.testing.do_bench(
        lambda: torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    )

    flops = 2 * N * out_h * out_w * K * C * R * S
    tflops_gluon = flops * 1e-12 / (ms_gluon * 1e-3)
    tflops_torch = flops * 1e-12 / (ms_torch * 1e-3)

    print(f"Gluon tcgen05 Latency: {ms_gluon:.3f} ms ({tflops_gluon:.2f} TFLOPS)")
    print(f"PyTorch Latency:      {ms_torch:.3f} ms ({tflops_torch:.2f} TFLOPS)")

