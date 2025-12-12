// -----// IR Dump Before GluonCanonicalize (gluon-canonicalize) ('builtin.module' operation) //----- //
#loc1 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":60:0)
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#loc14 = loc("in_desc"(#loc1))
#loc15 = loc("out_desc"(#loc1))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @memcpy_1d_tma_kernel(%in_desc: !tt.tensordesc<tensor<64xf32, #shared>> loc("in_desc"(#loc1)), %in_desc_0: i32 loc("in_desc"(#loc1)), %in_desc_1: i64 loc("in_desc"(#loc1)), %out_desc: !tt.tensordesc<tensor<64xf32, #shared>> loc("out_desc"(#loc1)), %out_desc_2: i32 loc("out_desc"(#loc1)), %out_desc_3: i64 loc("out_desc"(#loc1))) attributes {noinline = false} {
    %true = arith.constant true loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %pid = tt.get_program_id x : i32 loc(#loc16)
    %smem = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable> loc(#loc17)
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc18)
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc5)
    ttng.barrier_expect %bar, 256, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc6)
    %0 = arith.muli %pid, %c64_i32 : i32 loc(#loc7)
    ttng.async_tma_copy_global_to_local %in_desc[%0] %smem, %bar, %true : !tt.tensordesc<tensor<64xf32, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64xf32, #shared, #smem, mutable> loc(#loc8)
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc9)
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc10)
    ttng.async_tma_copy_local_to_global %out_desc[%0] %smem : !tt.tensordesc<tensor<64xf32, #shared>>, !ttg.memdesc<64xf32, #shared, #smem, mutable> loc(#loc11)
    ttng.async_tma_store_wait {pendings = 0 : i32} loc(#loc12)
    tt.return loc(#loc13)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":63:24)
#loc3 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":68:62)
#loc4 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":79:51)
#loc5 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":84:18)
#loc6 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":99:25)
#loc7 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":100:52)
#loc8 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":100:66)
#loc9 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":105:18)
#loc10 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":108:24)
#loc11 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":112:62)
#loc12 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":119:19)
#loc13 = loc("/home/scratch.bingyiz_gpu_2/projects/triton/Gluon-experimental/./test-tma1.py":119:4)
#loc16 = loc("pid"(#loc2))
#loc17 = loc("smem"(#loc3))
#loc18 = loc("bar"(#loc4))


