import os
import torch
import triton
import triton.language as tl

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_ALWAYS_COMPILE'] = '1'
os.environ['TRITON_KERNEL_DUMP'] = '1' 
os.environ['TRITON_DUMP_DIR'] = './kernel_dump'






# Exhaustive block-size/warp-stage sweep requested by user.
IMPLICIT_GEMM_AUTOTUNE_CONFIGS = [
    # triton.Config(
    #     {
    #         'BLOCK_M': block_m,
    #         'BLOCK_N': block_n,
    #         'BLOCK_K': block_k,
    #         'GROUP_SIZE_M': group_m,
    #     },
    #     num_warps=num_warps,
    #     num_stages=num_stages,
    # )
    # for block_m in (64, 128, 256)
    # for block_n in (64, 128, 256)
    # for block_k in (64, 128, 256, 512)
    # for group_m in (4, 8)
    # for num_warps in (4, 8)
    # for num_stages in (2, 3, 4)
    triton.Config(dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, GROUP_SIZE_M=4), num_warps=8, num_stages=3)
]

@triton.autotune(
    configs=IMPLICIT_GEMM_AUTOTUNE_CONFIGS,
    key=['N', 'H', 'W', 'C', 'K', 'R', 'S', 'out_h', 'out_w', 'stride_h', 'stride_w'],
)
@triton.jit
def implicit_gemm_conv2d_kernel(
    # Pointers
    input_ptr, weight_ptr, output_ptr,
    # Tensor Dimensions (Input: NHWC, Weight: KRSC, Output: NHWC)
    N, H, W, C,
    K, R, S, 
    out_h, out_w,
    # Strides (NHWC layout implies stride_c = 1, so we need others)
    stride_input_n, stride_input_h, stride_input_w,
    stride_weight_k, stride_weight_r, stride_weight_s,
    stride_output_n, stride_output_h, stride_output_w,
    # Conv parameters
    stride_h, stride_w,
    pad_h, pad_w,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Implicit GEMM Convolution Kernel (FP16)
    Input Layout:  NHWC
    Weight Layout: KRSC (Output_Channels, Filter_H, Filter_W, Input_Channels)
    Output Layout: NHWC
    """
    
    # 1. Program ID (Grid coordinates)
    pid = tl.program_id(axis=0)
    
    # 2. Define GEMM Dimensions
    # M = Batch * OutH * OutW (Total output pixels)
    # N = K (Output Channels)
    # K_GEMM = R * S * C (Filter volume)
    M_GEMM = N * out_h * out_w
    N_GEMM = K
    K_GEMM = R * S * C

    # 3. Tiling Logic (Standard Swizzled GEMM tiling)
    num_pid_m = tl.cdiv(M_GEMM, BLOCK_M)
    num_pid_n = tl.cdiv(N_GEMM, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 4. Coordinate Setup
    
    # Range of M (Output Pixels) for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Range of N (Output Channels) for this block
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Range of K (Accumulation) - initialized to 0
    offs_k = tl.arange(0, BLOCK_K)

    # -----------------------------------------------------------
    # IMPLICIT GEMM ADDRESS CALCULATION (The "Im2Col" on the fly)
    # -----------------------------------------------------------
    
    # De-linearize M index (offs_m) into (n, p, q) -> (Batch, OutY, OutX)
    # Note: doing division on ranges is valid in Triton
    # M = N * P * Q
    
    # batch_id = offs_m // (out_h * out_w)
    # rem_m = offs_m % (out_h * out_w)
    # out_y = rem_m // out_w
    # out_x = rem_m % out_w
    
    # Optimization: Pre-calculate common multipliers to avoid expensive division inside loop if possible.
    # However, for implicit gemm, we need these per-thread.
    
    batch_id = offs_m // (out_h * out_w)
    m_residual = offs_m % (out_h * out_w)
    out_y = m_residual // out_w
    out_x = m_residual % out_w

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 5. Main Loop over K_GEMM (Filter Volume)
    for k_block_start in range(0, K_GEMM, BLOCK_K):
        
        # Current K indices
        k_indices = k_block_start + offs_k
        
        # De-linearize K index into (r, s, c) -> (FilterY, FilterX, InChannel)
        # K_GEMM = R * S * C
        
        # We use the same logic as the pseudo-code
        # r = k_indices // (S * C)
        # rem_k = k_indices % (S * C)
        # s = rem_k // C
        # c = rem_k % C
        
        # Optimized math for Triton to reduce div/mod ops
        c_in = k_indices % C
        rem_rs = k_indices // C
        s_filter = rem_rs % S
        r_filter = rem_rs // S

        # -------------------------------------------------
        # CALCULATE INPUT POINTERS (A)
        # -------------------------------------------------
        
        # Calculate physical input coordinates based on convolution formula
        # in_y = out_y * stride + filter_y - padding
        curr_h = out_y[:, None] * stride_h + r_filter[None, :] - pad_h
        curr_w = out_x[:, None] * stride_w + s_filter[None, :] - pad_w
        
        # Create Mask for Padding (Zero Padding)
        # We need to check boundaries for H and W
        # broadcasting shapes: out_y is (BLOCK_M,), r_filter is (BLOCK_K,)
        # We need a 2D mask (BLOCK_M, BLOCK_K)
        
        # Triton broadcasts automatically:
        # curr_h becomes (BLOCK_M, BLOCK_K)
        mask_h = (curr_h >= 0) & (curr_h < H)
        mask_w = (curr_w >= 0) & (curr_w < W)
        mask_batch = batch_id < N # standard boundary check
        mask_k = k_indices < K_GEMM # standard loop boundary check
        
        load_mask = mask_h & mask_w & mask_batch[:, None] & mask_k[None, :]

        # Compute linear offset for Input Tensor (NHWC)
        # offset = n * stride_n + h * stride_h + w * stride_w + c
        # Note: We use stride_input_* passed from host
        input_offsets = (batch_id[:, None] * stride_input_n + 
                         curr_h * stride_input_h + 
                         curr_w * stride_input_w + 
                         c_in[None, :]) # stride_input_c is usually 1
        
        a_ptrs = input_ptr + input_offsets
        
        # Load Input Tile
        # If masked out (padding or out of bounds), load 0.0
        a_tile = tl.load(a_ptrs, mask=load_mask, other=0.0).to(tl.float16)

        # -------------------------------------------------
        # CALCULATE WEIGHT POINTERS (B)
        # -------------------------------------------------
        
        # Weight Layout: KRSC
        # GEMM B maps to (K_gemm, N_gemm) -> (R*S*C, OutChannels)
        # But strictly speaking for Gemm B: Row=K_index, Col=N_index
        # Layout is Linearized K on rows, Output Channels on Cols.
        
        # Weight Offset = k * stride_weight_k + n * stride_weight_n... 
        # Wait, our layout is KRSC. 
        # k (out_channel) is dimension 0. 
        # r, s, c are dimensions 1, 2, 3.
        # We are loading a tile of size (BLOCK_K, BLOCK_N).
        # BLOCK_K iterates over (r,s,c). BLOCK_N iterates over k (out_channels).
        
        # weight index logic:
        # We want weight[k_out, r, s, c]
        # k_out = offs_n
        # r, s, c comes from k_indices
        
        weight_offsets = (offs_n[None, :] * stride_weight_k + 
                          r_filter[:, None] * stride_weight_r + 
                          s_filter[:, None] * stride_weight_s + 
                          c_in[:, None]) # stride_weight_c is usually 1

        b_ptrs = weight_ptr + weight_offsets
        
        # Boundary check for weights (check N and K limits)
        weight_mask = (offs_n[None, :] < N_GEMM) & (k_indices[:, None] < K_GEMM)
        
        b_tile = tl.load(b_ptrs, mask=weight_mask, other=0.0).to(tl.float16)

        # -------------------------------------------------
        # MMA
        # -------------------------------------------------
        accumulator += tl.dot(a_tile, b_tile)

    # 6. Epilogue (Store Output)
    
    # Output Layout NHWC
    # M index -> (batch, out_y, out_x)
    # N index -> out_channel
    
    c_batch = offs_m // (out_h * out_w)
    c_rem = offs_m % (out_h * out_w)
    c_out_y = c_rem // out_w
    c_out_x = c_rem % out_w
    
    c_offsets = (c_batch[:, None] * stride_output_n + 
                 c_out_y[:, None] * stride_output_h + 
                 c_out_x[:, None] * stride_output_w + 
                 offs_n[None, :])
                 
    c_ptrs = output_ptr + c_offsets
    
    c_mask = (offs_m[:, None] < M_GEMM) & (offs_n[None, :] < N_GEMM)
    
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


# -----------------------------------------------------------------------------
# HOST WRAPPER
# -----------------------------------------------------------------------------

def implicit_gemm_conv2d(input_tensor, weight_tensor, stride=1, padding=0):
    """
    Args:
        input_tensor: (N, H, W, C)  <- NOTE: Channels Last
        weight_tensor: (K, R, S, C) <- NOTE: Channels Last
    """
    N, H, W, C = input_tensor.shape
    K, R, S, C_in = weight_tensor.shape
    assert C == C_in, "Input and Weight channels must match"
    
    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1
    
    output = torch.empty((N, out_h, out_w, K), device=input_tensor.device, dtype=torch.float16)
    
    # Grid Configuration
    M_GEMM = N * out_h * out_w
    N_GEMM = K
    
    grid = lambda META: (
        triton.cdiv(M_GEMM, META['BLOCK_M']) * triton.cdiv(N_GEMM, META['BLOCK_N']),
    )
    
    implicit_gemm_conv2d_kernel[grid](
        input_tensor, weight_tensor, output,
        N, H, W, C,
        K, R, S,
        out_h, out_w,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        weight_tensor.stride(0), weight_tensor.stride(1), weight_tensor.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        stride, stride,
        padding, padding,
    )
    
    return output

# -----------------------------------------------------------------------------
# VERIFICATION & BENCHMARK
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    
    # Shared parameters for verification and benchmarking
    N, H, W, C = 128, 64, 64, 384
    K, R, S = 384, 3, 3
    stride = 1
    padding = 1
    
    print(f"Parameters: N={N}, H={H}, W={W}, C={C}, K={K}, R={R}, S={S}, stride={stride}, padding={padding}")

    # Create Channels Last tensors for Tensor Core efficiency
    # 1. Input: (N, C, H, W) for PyTorch, but we permute to (N, H, W, C) for Triton
    x_nchw = torch.randn((N, C, H, W), device='cuda', dtype=torch.float16)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    
    # 2. Weights: (K, C, R, S) for PyTorch, permute to (K, R, S, C) for Triton
    w_nchw = torch.randn((K, C, R, S), device='cuda', dtype=torch.float16)
    w_krsc = w_nchw.permute(0, 2, 3, 1).contiguous()
    
    # -------------------------------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------------------------------
    print("\nRunning Verification...")
    
    torch.cuda.nvtx.range_push("Verification: Triton")
    # 3. Run Triton Kernel
    triton_out = implicit_gemm_conv2d(x_nhwc, w_krsc, stride=stride, padding=padding)
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("Verification: PyTorch")
    # 4. Run PyTorch Reference
    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    # Permute PyTorch output to match Triton (NHWC)
    torch_out = torch_out.permute(0, 2, 3, 1)

    torch.cuda.nvtx.range_pop()
    
    # 5. Compare
    print(f"Input Shape (NHWC): {x_nhwc.shape}")
    print(f"Output Shape (NHWC): {triton_out.shape}")
    
    if torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2):
        print("✅ Match!")
    else:
        print("❌ Mismatch")
        diff = (triton_out - torch_out).abs().max()
        print(f"Max diff: {diff}")

    # -------------------------------------------------------------------------
    # BENCHMARK
    # -------------------------------------------------------------------------
    print("\nRunning Benchmark...")

    # Warmup
    implicit_gemm_conv2d(x_nhwc, w_krsc, stride=stride, padding=padding)
    torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)

    # 1. Triton Benchmark
    # We measure the full Python wrapper call, but the heavy lifting is the kernel.
    torch.cuda.nvtx.range_push("Benchmark: Triton")
    ms_triton = triton.testing.do_bench(lambda: implicit_gemm_conv2d(x_nhwc, w_krsc, stride=stride, padding=padding))
    torch.cuda.nvtx.range_pop()
    
    # 2. PyTorch Benchmark
    # We measure only the kernel call, no permutations.
    torch.cuda.nvtx.range_push("Benchmark: PyTorch")
    ms_torch = triton.testing.do_bench(lambda: torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding))
    torch.cuda.nvtx.range_pop()
    
    print(f"Triton Latency:  {ms_triton:.3f} ms")
    print(f"PyTorch Latency: {ms_torch:.3f} ms")
    
    if ms_triton < ms_torch:
        print(f"Triton is {ms_torch / ms_triton:.2f}x FASTER than PyTorch")
    else:
        print(f"Triton is {ms_triton / ms_torch:.2f}x SLOWER than PyTorch")

