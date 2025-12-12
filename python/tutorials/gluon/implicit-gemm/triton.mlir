/home/scratch.bingyiz_gpu_2/projects/triton/python/tutorials/gluon/implicit-gemm/./implicit_gemm_conv2d_simple_gluon.py:198:12: error: cp.async does not support transfers smaller than 4 bytes; calculated this as 2 bytes
            weight_ptr + weight_offsets,
           ^
/home/scratch.bingyiz_gpu_2/projects/triton/python/tutorials/gluon/implicit-gemm/./implicit_gemm_conv2d_simple_gluon.py:198:12: error: failed to legalize operation 'ttg.async_copy_global_to_local' that was explicitly marked illegal
            weight_ptr + weight_offsets,
           ^
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @implicit_gemm_conv2d_gluon_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32, %arg9: i32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c255_i32 = arith.constant 255 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<1> : tensor<128x64xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x64xi32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %arg3, %arg10 : i32
    %2 = arith.muli %1, %arg11 : i32
    %3 = arith.muli %arg8, %arg9 : i32
    %4 = arith.muli %3, %arg6 : i32
    %5 = arith.addi %2, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.addi %arg7, %c255_i32 : i32
    %8 = arith.divsi %7, %c256_i32 : i32
    %9 = arith.muli %8, %c4_i32 : i32
    %10 = arith.divsi %0, %9 : i32
    %11 = arith.muli %10, %c4_i32 : i32
    %12 = arith.subi %6, %11 : i32
    %13 = arith.minsi %12, %c4_i32 : i32
    %14 = arith.remsi %0, %13 : i32
    %15 = arith.addi %11, %14 : i32
    %16 = arith.remsi %0, %9 : i32
    %17 = arith.divsi %16, %13 : i32
    %18 = arith.muli %15, %c128_i32 : i32
    %19 = arith.muli %17, %c256_i32 : i32
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = tt.splat %18 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.addi %21, %20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %23 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %24 = tt.splat %19 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %25 = arith.addi %24, %23 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %26 = arith.muli %arg10, %arg11 : i32
    %27 = tt.splat %26 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %28 = arith.divsi %22, %27 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %29 = arith.remsi %22, %27 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %30 = tt.splat %arg11 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %31 = arith.divsi %29, %30 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %32 = arith.remsi %29, %30 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %33 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %34 = ttg.local_alloc : () -> !ttg.memdesc<64x256xf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %35 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %36:2 = scf.for %arg21 = %c0_i32 to %4 step %c64_i32 iter_args(%arg22 = %c0_i32, %arg23 = %false) -> (i32, i1)  : i32 {
      %64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %65 = tt.splat %arg21 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %66 = arith.addi %65, %64 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %67 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = tt.splat %arg21 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = arith.addi %68, %67 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %70 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %71 = arith.remsi %66, %70 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %72 = arith.divsi %66, %70 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %73 = tt.splat %arg9 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %74 = arith.remsi %72, %73 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %75 = arith.divsi %72, %73 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %76 = tt.splat %arg6 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %77 = arith.remsi %69, %76 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %78 = arith.divsi %69, %76 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %79 = tt.splat %arg9 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %80 = arith.remsi %78, %79 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %81 = arith.divsi %78, %79 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %82 = tt.expand_dims %31 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %83 = tt.expand_dims %75 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
      %84 = tt.broadcast %82 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %85 = tt.broadcast %83 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %86 = arith.addi %84, %85 : tensor<128x64xi32, #blocked>
      %87 = arith.subi %86, %cst : tensor<128x64xi32, #blocked>
      %88 = tt.expand_dims %32 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %89 = tt.expand_dims %74 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
      %90 = tt.broadcast %88 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %91 = tt.broadcast %89 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %92 = arith.addi %90, %91 : tensor<128x64xi32, #blocked>
      %93 = arith.subi %92, %cst : tensor<128x64xi32, #blocked>
      %94 = arith.cmpi sge, %87, %cst_0 : tensor<128x64xi32, #blocked>
      %95 = tt.splat %arg4 : i32 -> tensor<128x64xi32, #blocked>
      %96 = arith.cmpi slt, %87, %95 : tensor<128x64xi32, #blocked>
      %97 = arith.andi %94, %96 : tensor<128x64xi1, #blocked>
      %98 = arith.cmpi sge, %93, %cst_0 : tensor<128x64xi32, #blocked>
      %99 = tt.splat %arg5 : i32 -> tensor<128x64xi32, #blocked>
      %100 = arith.cmpi slt, %93, %99 : tensor<128x64xi32, #blocked>
      %101 = arith.andi %98, %100 : tensor<128x64xi1, #blocked>
      %102 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %103 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked>
      %104 = arith.cmpi slt, %102, %103 : tensor<128x1xi32, #blocked>
      %105 = tt.expand_dims %66 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
      %106 = tt.splat %4 : i32 -> tensor<1x64xi32, #blocked>
      %107 = arith.cmpi slt, %105, %106 : tensor<1x64xi32, #blocked>
      %108 = arith.andi %97, %101 : tensor<128x64xi1, #blocked>
      %109 = tt.broadcast %104 : tensor<128x1xi1, #blocked> -> tensor<128x64xi1, #blocked>
      %110 = arith.andi %108, %109 : tensor<128x64xi1, #blocked>
      %111 = tt.broadcast %107 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
      %112 = arith.andi %110, %111 : tensor<128x64xi1, #blocked>
      %113 = tt.splat %arg12 : i32 -> tensor<128x1xi32, #blocked>
      %114 = arith.muli %102, %113 : tensor<128x1xi32, #blocked>
      %115 = tt.splat %arg13 : i32 -> tensor<128x64xi32, #blocked>
      %116 = arith.muli %87, %115 : tensor<128x64xi32, #blocked>
      %117 = tt.broadcast %114 : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %118 = arith.addi %117, %116 : tensor<128x64xi32, #blocked>
      %119 = tt.splat %arg14 : i32 -> tensor<128x64xi32, #blocked>
      %120 = arith.muli %93, %119 : tensor<128x64xi32, #blocked>
      %121 = arith.addi %118, %120 : tensor<128x64xi32, #blocked>
      %122 = tt.expand_dims %71 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
      %123 = tt.broadcast %122 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %124 = arith.addi %121, %123 : tensor<128x64xi32, #blocked>
      %125 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
      %126 = tt.addptr %125, %124 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %127 = ttg.async_copy_global_to_local %126, %33 mask %112 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #smem, mutable>
      %128 = tt.expand_dims %25 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %129 = tt.splat %arg15 : i32 -> tensor<1x256xi32, #blocked>
      %130 = arith.muli %128, %129 : tensor<1x256xi32, #blocked>
      %131 = tt.expand_dims %81 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %132 = tt.splat %arg16 : i32 -> tensor<64x1xi32, #blocked>
      %133 = arith.muli %131, %132 : tensor<64x1xi32, #blocked>
      %134 = tt.broadcast %130 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %135 = tt.broadcast %133 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %136 = arith.addi %134, %135 : tensor<64x256xi32, #blocked>
      %137 = tt.expand_dims %80 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %138 = tt.splat %arg17 : i32 -> tensor<64x1xi32, #blocked>
      %139 = arith.muli %137, %138 : tensor<64x1xi32, #blocked>
      %140 = tt.broadcast %139 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %141 = arith.addi %136, %140 : tensor<64x256xi32, #blocked>
      %142 = tt.expand_dims %77 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %143 = tt.broadcast %142 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %144 = arith.addi %141, %143 : tensor<64x256xi32, #blocked>
      %145 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
      %146 = arith.cmpi slt, %128, %145 : tensor<1x256xi32, #blocked>
      %147 = tt.expand_dims %69 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %148 = tt.splat %4 : i32 -> tensor<64x1xi32, #blocked>
      %149 = arith.cmpi slt, %147, %148 : tensor<64x1xi32, #blocked>
      %150 = tt.broadcast %146 : tensor<1x256xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %151 = tt.broadcast %149 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %152 = arith.andi %150, %151 : tensor<64x256xi1, #blocked>
      %153 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
      %154 = tt.addptr %153, %144 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %155 = ttg.async_copy_global_to_local %154, %34 mask %152 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared, #smem, mutable>
      %156 = ttg.async_commit_group
      %157 = ttg.async_wait {num = 0 : i32}
      ttng.fence_async_shared {bCluster = false}
      %158 = ttng.tc_gen5_mma %33, %34, %result[], %arg23, %true : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tc_gen5_commit %35 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %35, %arg22, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %159 = arith.xori %arg22, %c1_i32 : i32
      scf.yield %159, %true : i32, i1
    }
    ttng.inval_barrier %35 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result_1 = ttng.tmem_load %result : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
    %37 = arith.truncf %result_1 : tensor<128x256xf32, #linear> to tensor<128x256xf16, #linear>
    %38 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %39 = tt.splat %arg18 : i32 -> tensor<128x1xi32, #blocked>
    %40 = arith.muli %38, %39 : tensor<128x1xi32, #blocked>
    %41 = tt.expand_dims %31 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %42 = tt.splat %arg19 : i32 -> tensor<128x1xi32, #blocked>
    %43 = arith.muli %41, %42 : tensor<128x1xi32, #blocked>
    %44 = arith.addi %40, %43 : tensor<128x1xi32, #blocked>
    %45 = tt.expand_dims %32 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %46 = tt.splat %arg20 : i32 -> tensor<128x1xi32, #blocked>
    %47 = arith.muli %45, %46 : tensor<128x1xi32, #blocked>
    %48 = arith.addi %44, %47 : tensor<128x1xi32, #blocked>
    %49 = tt.expand_dims %25 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %50 = tt.broadcast %48 : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %51 = tt.broadcast %49 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %52 = arith.addi %50, %51 : tensor<128x256xi32, #blocked>
    %53 = tt.expand_dims %22 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %54 = tt.splat %2 : i32 -> tensor<128x1xi32, #blocked>
    %55 = arith.cmpi slt, %53, %54 : tensor<128x1xi32, #blocked>
    %56 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %57 = arith.cmpi slt, %49, %56 : tensor<1x256xi32, #blocked>
    %58 = tt.broadcast %55 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
    %59 = tt.broadcast %57 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
    %60 = arith.andi %58, %59 : tensor<128x256xi1, #blocked>
    %61 = ttg.convert_layout %37 : tensor<128x256xf16, #linear> -> tensor<128x256xf16, #blocked>
    %62 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    %63 = tt.addptr %62, %52 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
    tt.store %63, %61, %60 : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(tritongpu-combine-tensor-select-and-if, tritongpu-allocate-warp-groups, convert-scf-to-cf, gluon-inline, allocate-shared-memory-nv{compute-capability=100 ptx-version=88}, triton-tensor-memory-allocation, triton-nvidia-check-matmul-two-cta, tritongpu-global-scratch-memory-allocation, triton-nvidia-gpu-proxy-fence-insertion{compute-capability=100}, convert-triton-gpu-to-llvm{compute-capability=100 ptx-version=88}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, convert-nv-gpu-to-llvm, convert-warp-specialize-to-llvm, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce, convert-nvvm-to-llvm, enable-line-info)",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
<unknown>:0: error: Failures have been detected while processing an MLIR pass pipeline
<unknown>:0: note: Pipeline failed while executing [`ConvertTritonGPUToLLVM` on 'builtin.module' operation]: reproducer generated at `std::errs, please share the reproducer above with Triton project.`
Parameters: N=128, H=64, W=64, C=384, K=384, R=3, S=3, stride=1, padding=1
Device: NVIDIA B200
Compute Capability: (10, 0)

Running Verification...
Traceback (most recent call last):
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/tutorials/gluon/implicit-gemm/./implicit_gemm_conv2d_simple_gluon.py", line 342, in <module>
    gluon_out = implicit_gemm_conv2d_gluon(x_nhwc, w_krsc, stride=stride, padding=padding)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/tutorials/gluon/implicit-gemm/./implicit_gemm_conv2d_simple_gluon.py", line 288, in implicit_gemm_conv2d_gluon
    implicit_gemm_conv2d_gluon_kernel[grid](
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/triton/runtime/jit.py", line 370, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/triton/runtime/jit.py", line 720, in run
    kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/triton/runtime/jit.py", line 849, in _do_compile
    kernel = self.compile(src, target=target, options=options.__dict__)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/triton/compiler/compiler.py", line 324, in compile
    next_module = compile_ir(module, metadata)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/triton/backends/nvidia/compiler.py", line 544, in <lambda>
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.bingyiz_gpu_2/projects/triton/python/triton/backends/nvidia/compiler.py", line 378, in make_llir
    pm.run(mod, 'make_llir')
RuntimeError: PassManager::run failed
