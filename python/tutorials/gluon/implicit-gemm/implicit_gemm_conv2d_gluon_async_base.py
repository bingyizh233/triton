import os
import torch

import triton
import triton.language as tl
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

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_ALWAYS_COMPILE'] = '1'

# os.environ['TRITON_KERNEL_DUMP'] = '1' 
# os.environ['TRITON_DUMP_DIR'] = './kernel_dump_handwritten'

os.environ['TRITON_KERNEL_OVERRIDE'] = '1'
os.environ['TRITON_OVERRIDE_DIR'] = './kernel_dump_handwritten_overwrite'

## check if the target device is a Blackwell NVIDIA GPU
def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10



@gluon.jit
def implicit_gemm_conv2d_gluon_kernel(
    # Pointers
    input_ptr, weight_ptr, output_ptr,
    # Tensor Dimensions (Input: NHWC, Weight: KRSC, Output: NHWC)
    N, H, W, C: gl.constexpr,
    K, R: gl.constexpr, S: gl.constexpr,
    out_h: gl.constexpr, out_w: gl.constexpr,
    # Strides (NHWC layout)
    stride_input_n, stride_input_h, stride_input_w,
    stride_weight_k, stride_weight_r, stride_weight_s,
    stride_output_n, stride_output_h, stride_output_w,
    # Conv parameters. Not sure if we need to change it to gl.constexpr
    stride_h, stride_w,
    pad_h, pad_w,
    # Meta-parameters
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Implicit GEMM Convolution Kernel using Gluon (Blackwell tcgen05)
    
    Matches the logic of the provided Triton kernel and uses Blackwell-specific 
    instructions (tcgen05_mma, tmem) observed in the provided TTGIR.
    """

    # 1. Program ID and GEMM dimensions
    pid = gl.program_id(axis=0)
    
    M_GEMM = N * out_h * out_w
    N_GEMM = K
    K_GEMM = R * S * C

    # 2. Tiling Logic
    num_pid_m = gl.cdiv(M_GEMM, BLOCK_M)
    num_pid_n = gl.cdiv(N_GEMM, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m


    # 3. Coordinate Setup
    # Layouts matching TTGIR #blocked encodings
    # Using 4 elements per thread (8 bytes) to be safe with vectorization
    A_VEC: gl.constexpr = 8
    B_VEC: gl.constexpr = 8

    # This is the layout for A in the optimized triton kernel
    a_layout: gl.constexpr = gl.BlockedLayout([1, A_VEC], [4, 8], [num_warps, 1], [1, 0])
    a_cols_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=a_layout)
    a_rows_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=a_layout)

    # This is the layout for B in the optimized triton kernel
    b_layout: gl.constexpr = gl.BlockedLayout([B_VEC, 1], [8, 4], [1, num_warps], [0, 1])
    b_cols_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=b_layout)
    b_rows_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=b_layout)

    # This is the layout for output in the optimized triton kernel
    c_layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [num_warps, 1], [1, 0])
    c_cols_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=c_layout)
    c_rows_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=c_layout)

    offs_m_base = pid_m * BLOCK_M
    offs_n_base = pid_n * BLOCK_N

    offs_m = offs_m_base + gl.arange(0, BLOCK_M, layout=a_rows_layout)
    offs_n = offs_n_base + gl.arange(0, BLOCK_N, layout=b_cols_layout)
    
    # Batch_id, m_residual, out_y, out_x are in M dimension
    # offs_m is the slice layout in M dimension
    batch_id = offs_m // (out_h * out_w)
    m_residual = offs_m % (out_h * out_w)
    out_y = m_residual // out_w
    out_x = m_residual % out_w

    # Allocate Memory
    # Shared memory for Async Copy
    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16, transposed=True) ## This is the tracky part, need to transpose the layout manually

    a_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_M, BLOCK_K], a_smem_layout)
    b_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_K, BLOCK_N], b_smem_layout)

     # Tensor Memory for Accumulator (Blackwell)
    # Matches #tmem: blockM=128, blockN=256
    tmem_layout: gl.constexpr = TensorMemoryLayout(
        block=(128, BLOCK_N),
        col_stride=1,
    )
    acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], tmem_layout)

    # Barrier for MMA
    mma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(mma_bar, count=1)
    phase = 0

    # use_acc=False for first iteration to zero-initialize accumulator
    use_acc = False

     # 5. Main Loop
    for k_block_start in range(0, K_GEMM, BLOCK_K):
        # K indices
        offs_k_col = k_block_start + gl.arange(0, BLOCK_K, layout=a_cols_layout)
        offs_k_row = k_block_start + gl.arange(0, BLOCK_K, layout=b_rows_layout)
        
        # De-linearize K for A (K is cols)
        c_in_col = offs_k_col % C
        rem_rs_col = offs_k_col // C
        s_filter_col = rem_rs_col % S
        r_filter_col = rem_rs_col // S

        # De-linearize K for B (K is rows)
        c_in_row = offs_k_row % C
        rem_rs_row = offs_k_row // C
        s_filter_row = rem_rs_row % S
        r_filter_row = rem_rs_row // S

        # --- Load A ---
        # out_y is slice layout of A in M dimension
        # r_filter_col is slice layout of A in K dimension
        curr_h = out_y[:, None] * stride_h + r_filter_col[None, :] - pad_h

        # out_x is slice layout of W in M dimension
        # s_filter_col is slice layout of A in K dimemsion
        curr_w = out_x[:, None] * stride_w + s_filter_col[None, :] - pad_w

        mask_h = (curr_h >= 0) & (curr_h < H)
        mask_w = (curr_w >= 0) & (curr_w < W)
        mask_batch = batch_id < N # slice layout in M dimesion
        mask_k_a = offs_k_col < K_GEMM  # slice layout in K dimension

        load_mask_a = mask_h & mask_w & mask_batch[:, None] & mask_k_a[None, :]

        # Compute linear offset for Input Tensor (NHWC)
        input_offsets = (batch_id[:, None] * stride_input_n +         
                         curr_h * stride_input_h + 
                         curr_w * stride_input_w + 
                         c_in_col[None, :])
        
        # input_offsets = tl.multiple_of(input_offsets, 8)
        
        cp.async_copy_global_to_shared(a_smem, input_ptr + input_offsets, mask=load_mask_a)

        
        weight_offsets = (offs_n[None, :] * stride_weight_k + 
                          r_filter_row[:, None] * stride_weight_r + 
                          s_filter_row[:, None] * stride_weight_s + 
                          c_in_row[:, None])

        weight_mask = (offs_n[None, :] < N_GEMM) & (offs_k_row[:, None] < K_GEMM)
        
        # weight_offsets = tl.multiple_of(weight_offsets, 8)

        cp.async_copy_global_to_shared(b_smem, weight_ptr + weight_offsets, mask=weight_mask)
        
        cp.commit_group()
        cp.wait_group(0)
        fence_async_shared()

                # --- MMA ---
        tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=use_acc)
        tcgen05_commit(mma_bar)
        mbarrier.wait(mma_bar, phase=phase)
        
        use_acc = True
        phase ^= 1
    mbarrier.invalidate(mma_bar)

    # 6. Epilogue

    acc_reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (256, BLOCK_N), tmem_layout, num_warps
    )
    
    acc = acc_tmem.load(acc_reg_layout)
    acc_fp16 = acc.to(gl.float16)
    acc_fp16 = gl.convert_layout(acc_fp16, c_layout)

    c_offs_m = offs_m_base + gl.arange(0, BLOCK_M, layout=c_rows_layout)
    c_offs_n = offs_n_base + gl.arange(0, BLOCK_N, layout=c_cols_layout)

    c_batch = c_offs_m // (out_h * out_w)
    c_rem = c_offs_m % (out_h * out_w)
    c_out_y = c_rem // out_w
    c_out_x = c_rem % out_w

    c_offsets = (c_batch[:, None] * stride_output_n + 
        c_out_y[:, None] * stride_output_h + 
        c_out_x[:, None] * stride_output_w + 
        c_offs_n[None, :])

    c_mask = (c_offs_m[:, None] < M_GEMM) & (c_offs_n[None, :] < N_GEMM)
    gl.store(output_ptr + c_offsets, acc_fp16, mask=c_mask)






def implicit_gemm_conv2d_gluon(input_tensor, weight_tensor, stride=1, padding=0):
    """
    Args:
        input_tensor: (N, H, W, C)  <- NOTE: Channels Last
        weight_tensor: (K, R, S, C) <- NOTE: Channels Last
    """

    if not is_blackwell():
        raise RuntimeError("This kernel requires a Blackwell NVIDIA GPU (SM 10.x)")
    
    # Get the shape of the input and weight tensors
    
    # Shared parameters for verification and benchmarking
    # N is the number of batches
    # H is the height of the input feature map
    # W is the width of the input feature map
    # C is the number of input channels
    # K is the nubmer of output channels
    # R is the height of the filter 
    # S is the width of the filter
    # stride and padding
    N, H, W, C = input_tensor.shape
    K, R, S, C_in = weight_tensor.shape
    assert C == C_in, "Input and Weight channels must match"

    # Calculate the output height and width
    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1

    # N is the batch size, K is the number of output channels
    output = torch.empty((N, out_h, out_w, K), device=input_tensor.device, dtype=torch.float16)

    
    # Grid Configuration
    # This is the shape of the output tensor
    M_GEMM = N * out_h * out_w
    N_GEMM = K


    # Tuning parameters - chosen for tcgen05 MMA efficiency
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 64
    GROUP_SIZE_M = 4
    num_warps = 8


    # The grid size is chosen based on the number of output pixels and the number of output channels
    grid = (
        triton.cdiv(M_GEMM, BLOCK_M) * triton.cdiv(N_GEMM, BLOCK_N),
    )



    implicit_gemm_conv2d_gluon_kernel[grid](
        input_tensor, weight_tensor, output,
        N, H, W, C,
        K, R, S,
        out_h, out_w,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2),
        weight_tensor.stride(0), weight_tensor.stride(1), weight_tensor.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        stride, stride,
        padding, padding,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )
    
    return output





if __name__ == "__main__":
    if not is_blackwell():
        print("This tutorial requires a Blackwell NVIDIA GPU (SM 10.x)")
        print("Exiting...")
        exit(0)
    
    torch.manual_seed(0)

    # Shared parameters for verification and benchmarking
    # N is the number of batches
    # H is the height of the input feature map
    # W is the width of the input feature map
    # C is the number of input channels
    # K is the nubmer of output channels
    # R is the height of the filter 
    # S is the width of the filter
    # stride and padding
    N, H, W, C = 128, 64, 64, 384
    K, R, S = 384, 3, 3
    stride = 1
    padding = 1

    print(f"Parameters: N={N}, H={H}, W={W}, C={C}, K={K}, R={R}, S={S}, stride={stride}, padding={padding}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")


    # Create Channels Last tensors for Tensor Core efficiency
    # 1. Input: (N, C, H, W) for PyTorch, but we permute to (N, H, W, C) for Triton
    # (N, H, W, C) is for im2col aglorithm
    # For now, we use torch.float16 for the input and weight tensors
    x_nchw = torch.randn((N, C, H, W), device='cuda', dtype=torch.float16)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()


    # 2. Weights: (K, R, S, C) for PyTorch, permute to (K, R, S, C) for Triton
    # (K, R, S, C) is for im2col aglorithm
    # For now, we use torch.float16 for the input and weight tensors
    w_nchw = torch.randn((K, C, R, S), device='cuda', dtype=torch.float16)
    w_krsc = w_nchw.permute(0, 2, 3, 1).contiguous()


    # -------------------------------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------------------------------
    print("\nRunning Verification...")
    
    torch.cuda.nvtx.range_push("Verification: Triton")
    # 3. Run Triton Kernel
    triton_out = implicit_gemm_conv2d_gluon(x_nhwc, w_krsc, stride=stride, padding=padding)
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
    for _ in range(5):
        implicit_gemm_conv2d_gluon(x_nhwc, w_krsc, stride=stride, padding=padding)
        torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch.cuda.synchronize()

    # 1. Gluon tcgen05 Benchmark
    torch.cuda.nvtx.range_push("Benchmark: Gluon tcgen05")
    ms_gluon = triton.testing.do_bench(
        lambda: implicit_gemm_conv2d_gluon(x_nhwc, w_krsc, stride=stride, padding=padding),
        warmup=100, rep=500
    )
    torch.cuda.nvtx.range_pop()
    
    # 2. PyTorch Benchmark
    torch.cuda.nvtx.range_push("Benchmark: PyTorch")
    ms_torch = triton.testing.do_bench(
        lambda: torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding),
        warmup=100, rep=500
    )
    torch.cuda.nvtx.range_pop()
    
    # Calculate TFLOPS
    # Conv2D FLOPs = 2 * N * out_h * out_w * K * C * R * S
    out_h = (H + 2 * padding - R) // stride + 1
    out_w = (W + 2 * padding - S) // stride + 1
    flops = 2 * N * out_h * out_w * K * C * R * S
    
    tflops_gluon = flops * 1e-12 / (ms_gluon * 1e-3)
    tflops_torch = flops * 1e-12 / (ms_torch * 1e-3)
    
    print(f"\nGluon tcgen05 Latency:  {ms_gluon:.3f} ms ({tflops_gluon:.2f} TFLOPS)")
    print(f"PyTorch Latency:        {ms_torch:.3f} ms ({tflops_torch:.2f} TFLOPS)")
    
    if ms_gluon < ms_torch:
        print(f"\nGluon tcgen05 is {ms_torch / ms_gluon:.2f}x FASTER than PyTorch")
    else:
        print(f"\nGluon tcgen05 is {ms_gluon / ms_torch:.2f}x SLOWER than PyTorch")


