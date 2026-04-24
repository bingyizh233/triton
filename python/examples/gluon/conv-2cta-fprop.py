"""Warp-specialized 2-CTA forward convolution (implicit GEMM, NHWC) -- V4.

Four warp_specialize partitions: epilogue / MMA / load / CLC scheduler.
The CLC partition issues ``clc.try_cancel`` to hand additional output tiles
to the three consumer partitions via a shared-memory planar-pid slot ring;
each kernel launch starts on the tile matching its ``gl.program_id`` and
loops until CLC reports no more work. Cluster MMA uses ``multicast=True``,
per-CTA TMA loads stream through a shared-memory ring (STAGES buffers),
accumulation lives in a one-deep TMEM ring with ``cga_layout=((1,0),),
two_ctas=True``, and the epilogue TMA-stores each tile in N-subtiles.

Compiler gotcha (2026-04-23): calling ``cluster.cluster_cta_id()`` *inside*
a warp-specialized partition body causes that partition to be extracted
into a separate ``.func`` by the gluon lowering pass, which produces a
silent miscompile that deadlocks on ``mbarrier.try_wait`` of
``load_ready_bars``. Always read ``cluster_cta_id()`` in the kernel entry
and pass it through the partition args (``V4Args.cid``).

Cluster contract:
    CGA_LAYOUT = ((1, 0),)                       # 2-CTA M-split
    CTA_M      = block_size_m                    # per-CTA M tile
    CTA_N      = block_size_n // 2               # per-CTA N (B is always N-split in cluster)
    TILE_M     = 2 * CTA_M                       # cluster-wide output M
    TILE_N     = block_size_n                    # cluster-wide output N
    BLOCK_K    = block_size_k                    # K reduction tile (not split)

Verified performance on B200, bf16, N=128, 64x64, 3x3, stride=1, pad=1,
with the CLC scheduler + ``acc_stages=2`` TMEM ring:
    Co=384, bm=128 bn=128 bk=128 epi=128 stages=4: 1400.8 TFLOPS
    Co=512, bm=128 bn=256 bk=64  epi=256 stages=5: 1453.3 TFLOPS (peak)

Recommended config:
    ``acc_stages=2`` is ~7-15% faster than ``acc_stages=1`` on the target
    shapes by overlapping MMA on tile N+1 with epilogue writing tile N.
    For Co multiples of 256, use ``block_size_n=256 block_size_k=64
    stages=5``. For smaller Co tails (e.g. Co=384) prefer
    ``block_size_n=128 block_size_k=128 stages=4`` -- the smaller N tile
    lines up with Co and avoids the narrower N-subtile epilogue.

Requires: Blackwell GPU (SM 10.x), num_ctas=2 cluster launch.
"""

import pytest
import torch

import triton
import triton.language as tl

from triton.language.core import _aggregate as aggregate

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor, TensorDescriptorIm2Col
from triton.experimental.gluon.language.nvidia.hopper import cluster, mbarrier, tma
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    clc,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
    tensor_memory_descriptor,
)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return torch.cuda.is_available() and is_cuda() and torch.cuda.get_device_capability()[0] == 10


TORCH_GEMM_DTYPE = torch.bfloat16
GL_GEMM_DTYPE = gl.bfloat16


def _set_default_tma_allocator(device):
    def alloc_fn(size: int, alignment: int, stream):
        del alignment, stream
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


@gluon.constexpr_function
def get_split_dim(cga_layout, dim):
    return 1 << sum(basis[dim] != 0 for basis in cga_layout)


@gluon.constexpr_function
def get_broadcast_cga_layout(cga_layout):
    return tuple(tuple(0 for _ in basis) for basis in cga_layout)


@gluon.constexpr_function
def _get_operand_cga_layout(layout, op_idx):
    assert op_idx in (0, 1)
    if not layout:
        return layout
    assert layout[0] == (1, 0)
    first = (1, 0) if op_idx == 0 else (0, 1)

    def broadcast(basis):
        return (basis[0], 0) if op_idx == 0 else (0, 2 * basis[1])

    return (first, *map(broadcast, layout[1:]))


def _validate_cga_layout(cga_layout):
    if len(cga_layout) != 1 or cga_layout[0] != (1, 0):
        raise ValueError(f"Only 2-CTA M-split ``((1, 0),)`` is supported, got {cga_layout!r}")


def normalize_2d(value, name):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} must be an int or length-2 tuple/list, got {value!r}")


def maybe_pad_ci_for_tma(input_tensor, weight_tensor, alignment_bytes=16):
    """Pad NHWC/OHWI channel dimensions so TMA-visible strides are 16B aligned."""
    input_elem_bytes = input_tensor.element_size()
    weight_elem_bytes = weight_tensor.element_size()
    if input_elem_bytes != weight_elem_bytes:
        raise ValueError("Input and weight element sizes must match")
    if alignment_bytes % input_elem_bytes != 0:
        raise ValueError(
            f"alignment_bytes={alignment_bytes} must be divisible by element size {input_elem_bytes}"
        )

    ci_alignment = alignment_bytes // input_elem_bytes
    orig_ci = input_tensor.shape[-1]
    padded_ci = triton.cdiv(orig_ci, ci_alignment) * ci_alignment
    if padded_ci == orig_ci:
        return input_tensor, weight_tensor

    padded_input = input_tensor.new_zeros((*input_tensor.shape[:-1], padded_ci))
    padded_input[..., :orig_ci] = input_tensor

    padded_weight = weight_tensor.new_zeros((*weight_tensor.shape[:-1], padded_ci))
    padded_weight[..., :orig_ci] = weight_tensor

    return padded_input.contiguous(), padded_weight.contiguous()


# ===-----------------------------------------------------------------------===#
# Kernel helpers
# ===-----------------------------------------------------------------------===#


@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)


@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor

    @gluon.jit
    def initialize(num_tiles):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        pid_per_kernel = gl.cdiv(num_tiles, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = gl.minimum(pid_start + pid_per_kernel, num_tiles)
        return PersistentTileScheduler(pid_start, pid_end)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile_id(self, idx):
        return self.pid_start + idx


@aggregate
class V4Config:
    Co: gl.tensor
    R: gl.tensor
    S: gl.tensor
    out_h: gl.tensor
    out_w: gl.tensor
    stride_h: gl.tensor
    stride_w: gl.tensor
    pad_h: gl.tensor
    pad_w: gl.tensor
    Ci: gl.tensor
    M_GEMM: gl.tensor

    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    CTA_M: gl.constexpr
    CTA_N: gl.constexpr
    BLOCK_K: gl.constexpr
    GROUP_SIZE_M: gl.constexpr

    @gluon.jit
    def get_program(self, pid):
        num_pid_n = gl.cdiv(self.Co, self.TILE_N)
        num_pid_m = gl.cdiv(self.M_GEMM, self.TILE_M)
        num_pid_in_group = self.GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * self.GROUP_SIZE_M
        group_size_m = gl.minimum(num_pid_m - first_pid_m, self.GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        return V4Program(self, pid_m, pid_n)

    @gluon.jit
    def get_num_k_iterations(self):
        return self.R * self.S * gl.cdiv(self.Ci, self.BLOCK_K)

    @gluon.jit
    def get_num_tiles(self):
        return gl.cdiv(self.M_GEMM, self.TILE_M) * gl.cdiv(self.Co, self.TILE_N)


@aggregate
class V4Program:
    config: V4Config
    pid_m: gl.tensor
    pid_n: gl.tensor

    @gluon.jit
    def get_cluster_offsets(self):
        return self.pid_m * self.config.TILE_M, self.pid_n * self.config.TILE_N

    @gluon.jit
    def get_cta_offsets(self, cid):
        off_m, off_n = self.get_cluster_offsets()
        return off_m + cid * self.config.CTA_M, off_n + cid * self.config.CTA_N

    @gluon.jit
    def get_m_offsets(self, cid):
        cta_off_m, _ = self.get_cta_offsets(cid)
        out_x = cta_off_m % self.config.out_w
        out_y = (cta_off_m // self.config.out_w) % self.config.out_h
        batch_id = (cta_off_m // self.config.out_w) // self.config.out_h
        return batch_id, out_y, out_x


@aggregate
class ClcTileSchedulerConsumer:
    has_work: gl.tensor
    tile_id: gl.tensor
    pid_m: gl.tensor
    pid_n: gl.tensor
    config: V4Config
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    counter: Counter
    consumed_counter: Counter

    @gluon.jit
    def initialize(config, clc_result_buffers, clc_barriers,
                   clc_planar_pid_buffers, clc_planar_ready_bars, clc_consumed_bars):
        tile_id = gl.program_id(axis=0)
        prog = config.get_program(tile_id)
        has_work = gl.to_tensor(True)
        counter = Counter.create(0, clc_barriers.shape[0])
        consumed_counter = Counter.create(0, clc_barriers.shape[0])
        return ClcTileSchedulerConsumer(
            has_work, tile_id, prog.pid_m, prog.pid_n, config,
            clc_result_buffers, clc_barriers, clc_planar_pid_buffers,
            clc_planar_ready_bars, clc_consumed_bars,
            counter, consumed_counter,
        )

    @gluon.jit
    def step(self, iteration):
        consumed_counter = self.consumed_counter
        if iteration > 0:
            mbarrier.arrive(self.clc_consumed_bars.index(consumed_counter.index))
            consumed_counter = consumed_counter.next()
        counter = self.counter
        barrier = self.clc_barriers.index(counter.index)
        result = self.clc_result_buffers.index(counter.index)
        mbarrier.wait(barrier, counter.phase)
        clc_res = clc.load_result(result)
        mbarrier.wait(self.clc_planar_ready_bars.index(counter.index), counter.phase)
        planar_slot = self.clc_planar_pid_buffers.index(counter.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        packed_pid = planar_slot.load(planar_layout).reshape([])
        pid_m = ((packed_pid >> 32) & 0xFFFFFFFF).to(gl.int32)
        pid_n = (packed_pid & 0xFFFFFFFF).to(gl.int32)
        has_work = clc_res.is_canceled()
        tile_id = self.tile_id
        if has_work:
            tile_id = clc_res.program_id(0)
        return ClcTileSchedulerConsumer(
            has_work, tile_id, pid_m, pid_n, self.config,
            self.clc_result_buffers, self.clc_barriers,
            self.clc_planar_pid_buffers, self.clc_planar_ready_bars,
            self.clc_consumed_bars,
            counter.next(), consumed_counter,
        )


@aggregate
class V4Args:
    config: V4Config
    cid: gl.tensor
    a_desc: tma.tensor_descriptor_im2col
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    acc_bufs: tensor_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_planar_pid_buffers: gl.shared_memory_descriptor
    clc_planar_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor

    @gluon.jit
    def get_clc_consumer(self):
        return ClcTileSchedulerConsumer.initialize(
            self.config,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_planar_pid_buffers,
            self.clc_planar_ready_bars,
            self.clc_consumed_bars,
        )


# ===-----------------------------------------------------------------------===#
# Warp-specialized partitions
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _v4_load(p):
    """Producer: per-CTA TMA loads into the cluster-partitioned SMEM ring.

    Each CTA loads ``[CTA_M, BLOCK_K]`` of A (M-split) and
    ``[BLOCK_K, CTA_N]`` of B (N-split) per iteration. ``load_ready_bars``
    is two_ctas=True so MMA waits for both CTAs' TMAs to complete.
    """
    a_desc = p.a_desc
    b_desc = p.b_desc
    CTA_M: gl.constexpr = a_desc.block_shape[0]
    CTA_N: gl.constexpr = b_desc.block_shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_shape[1]
    local_cga_layout: gl.constexpr = get_broadcast_cga_layout(p.c_desc.layout.cga_layout)
    a_tma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [CTA_M, BLOCK_K], a_desc.dtype, cga_layout=local_cga_layout,
    )
    b_tma_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, CTA_N], b_desc.dtype, cga_layout=local_cga_layout,
    )

    config = p.config
    cid = p.cid
    num_k_iter = config.get_num_k_iterations()
    STAGES: gl.constexpr = p.load_empty_bars.shape[0]
    state = Counter.create(1, STAGES)
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        prog = V4Program(config, scheduler.pid_m, scheduler.pid_n)
        batch_id, out_y, out_x = prog.get_m_offsets(cid)
        _, cta_off_n = prog.get_cta_offsets(cid)
        for k_iter in range(num_k_iter):
            a_stage = p.a_bufs.index(state.index)
            b_stage = p.b_bufs.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase, deps=[a_stage, b_stage])
            a_stage_local = a_stage._reinterpret(a_desc.dtype, [CTA_M, BLOCK_K], a_tma_layout)
            b_stage_local = b_stage._reinterpret(b_desc.dtype, [BLOCK_K, CTA_N], b_tma_layout)

            iter_ci = k_iter // (config.R * config.S)
            remain_rs = k_iter % (config.R * config.S)
            iter_s = remain_rs % config.S
            iter_r = remain_rs // config.S

            bar = p.load_ready_bars.index(state.index)
            mbarrier.expect(bar, a_desc.nbytes_per_cta + b_desc.nbytes_per_cta)
            tma.async_copy_global_to_shared_im2col(
                a_desc,
                [
                    batch_id,
                    out_y * config.stride_h - config.pad_h,
                    out_x * config.stride_w - config.pad_w,
                    iter_ci * BLOCK_K,
                ],
                [iter_r.to(tl.int16), iter_s.to(tl.int16)],
                bar,
                a_stage_local,
            )
            k_offset = (iter_r * config.S + iter_s) * config.Ci + iter_ci * BLOCK_K
            tma.async_copy_global_to_shared(b_desc, [k_offset, cta_off_n], bar, b_stage_local)
            state = state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def _v4_mma(p):
    """Consumer: cluster MMA over the ring. Each MMA's inline
    ``mbarriers=[load_empty[i]]`` releases the buffer back to the load
    partition. After the K-loop, ``tcgen05_commit(acc_ready, descs=[a,b])``
    drains pending async MMAs before signalling the epilogue.
    """
    config = p.config
    num_k_iter = config.get_num_k_iterations()
    STAGES: gl.constexpr = p.load_empty_bars.shape[0]
    load_state = Counter.create(0, STAGES)
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        use_acc = False
        for _k_iter in range(num_k_iter):
            a_stage = p.a_bufs.index(load_state.index)
            b_stage = p.b_bufs.index(load_state.index)
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase, deps=[a_stage, b_stage])
            tcgen05_mma(
                a_stage, b_stage, acc_buf,
                use_acc=use_acc,
                multicast=True,
                mbarriers=[p.load_empty_bars.index(load_state.index)],
            )
            load_state = load_state.next()
            use_acc = True
        tcgen05_commit(p.acc_ready_bars.index(acc_state.index))
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1


@gluon.jit
def _v4_epilogue(p):
    """Waits for acc_ready, slices the TMEM accumulator, and TMA-stores
    N-subtiles using a narrower epilogue descriptor."""
    TILE_M: gl.constexpr = p.config.TILE_M
    TILE_N: gl.constexpr = p.config.TILE_N
    EPILOGUE_BLOCK_N: gl.constexpr = p.c_desc.block_shape[1]
    gl.static_assert(TILE_N % EPILOGUE_BLOCK_N == 0)
    SUBTILE_FACTOR: gl.constexpr = TILE_N // EPILOGUE_BLOCK_N
    SUBTILE_STAGES: gl.constexpr = 1 if SUBTILE_FACTOR == 1 else 2
    acc_smems = gl.allocate_shared_memory(
        p.c_desc.dtype,
        [SUBTILE_STAGES, TILE_M, EPILOGUE_BLOCK_N],
        p.c_desc.layout,
    )
    sub_state = Counter.create(0, SUBTILE_STAGES)
    acc_state = Counter.create(0, p.acc_empty_bars.shape[0])
    config = p.config
    scheduler = p.get_clc_consumer()
    i = 0
    while scheduler.has_work:
        prog = V4Program(config, scheduler.pid_m, scheduler.pid_n)
        off_m, off_n = prog.get_cluster_offsets()
        mbarrier.wait(p.acc_ready_bars.index(acc_state.index), acc_state.phase)
        acc_buf = p.acc_bufs.index(acc_state.index)
        for s in gl.static_range(SUBTILE_FACTOR):
            acc_sub = acc_buf.slice(EPILOGUE_BLOCK_N * s, EPILOGUE_BLOCK_N)
            acc_smem = acc_smems.index(sub_state.index)
            acc_tile = acc_sub.load().to(p.c_desc.dtype)
            tma.store_wait(pendings=SUBTILE_STAGES - 1)
            acc_smem.store(acc_tile)
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + EPILOGUE_BLOCK_N * s], acc_smem)
            sub_state = sub_state.next()
        mbarrier.arrive(p.acc_empty_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()
        scheduler = scheduler.step(i)
        i += 1
    tma.store_wait(0)


@gluon.jit
def _v4_clc(p):
    """CLC scheduler partition: pulls work tile ids from the hardware
    cluster-launch-control unit and broadcasts the resolved (pid_m, pid_n)
    to the consumer partitions via a shared-memory planar-pid slot ring."""
    has_work = gl.to_tensor(True)
    state = Counter.create(0, p.clc_barriers.shape[0])
    consumed_state = Counter.create(1, p.clc_barriers.shape[0])
    ACC_STAGES: gl.constexpr = p.clc_barriers.shape[0]
    i = 0
    while has_work:
        # Reuse the slot only after all consumer partitions signaled consumed.
        mbarrier.wait(p.clc_consumed_bars.index(consumed_state.index), consumed_state.phase, pred=(i >= ACC_STAGES))
        barrier = p.clc_barriers.index(state.index)
        result = p.clc_result_buffers.index(state.index)
        # 16: clc.try_cancel uses a `.b128` result payload.
        mbarrier.expect(barrier, 16)
        clc.try_cancel(result, barrier)
        mbarrier.wait(barrier, state.phase)
        clc_res = clc.load_result(result)
        has_work = clc_res.is_canceled()
        pid_m = gl.to_tensor(0)
        pid_n = gl.to_tensor(0)
        if has_work:
            tile_id = clc_res.program_id(0)
            prog = p.config.get_program(tile_id)
            pid_m = prog.pid_m
            pid_n = prog.pid_n
        packed_pid = (pid_m.to(gl.int64) << 32) | (pid_n.to(gl.int64) & 0xFFFFFFFF)
        planar_slot = p.clc_planar_pid_buffers.index(state.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        planar_slot.store(gl.full([1], packed_pid, gl.int64, layout=planar_layout))
        mbarrier.arrive(p.clc_planar_ready_bars.index(state.index))
        state = state.next()
        consumed_state = consumed_state.next()
        i += 1


# ===-----------------------------------------------------------------------===#
# Kernel entry point
# ===-----------------------------------------------------------------------===#


@gluon.jit(do_not_specialize=[
    "N", "H", "W", "R", "S", "pad_h", "pad_w",
])
def _conv2d_im2col_2cta_ws_v4_kernel(
    a_desc, b_desc, c_desc,
    N, H, W, Ci, Co, R, S,
    out_h, out_w,
    stride_h, stride_w, pad_h, pad_w,
    GROUP_SIZE_M: gl.constexpr,
    STAGES: gl.constexpr,
    ACC_STAGES: gl.constexpr,
):
    """V4: warp-specialized, 3 partitions (epilogue, load, MMA), ring
    buffers, multicast cluster MMA, TMA-store epilogue. No CLC."""
    gl.static_assert(gl.num_ctas() == 2)
    CTA_M: gl.constexpr = a_desc.block_shape[0]
    CTA_N: gl.constexpr = b_desc.block_shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_shape[1]
    TILE_M: gl.constexpr = c_desc.block_shape[0]
    EPILOGUE_BLOCK_N: gl.constexpr = c_desc.block_shape[1]
    cga_layout: gl.constexpr = c_desc.layout.cga_layout
    a_cga_layout: gl.constexpr = _get_operand_cga_layout(cga_layout, 0)
    b_cga_layout: gl.constexpr = _get_operand_cga_layout(cga_layout, 1)
    TILE_N: gl.constexpr = CTA_N * get_split_dim(b_cga_layout, 1)

    gl.static_assert(get_split_dim(cga_layout, 0) == gl.num_ctas())
    gl.static_assert(TILE_M == CTA_M * 2)
    gl.static_assert(c_desc.block_shape[0] == TILE_M)
    gl.static_assert(TILE_N % EPILOGUE_BLOCK_N == 0)

    M_GEMM = N * out_h * out_w

    config = V4Config(
        gl.to_tensor(Co), gl.to_tensor(R), gl.to_tensor(S),
        gl.to_tensor(out_h), gl.to_tensor(out_w),
        gl.to_tensor(stride_h), gl.to_tensor(stride_w),
        gl.to_tensor(pad_h), gl.to_tensor(pad_w),
        gl.to_tensor(Ci), M_GEMM,
        TILE_M, TILE_N, CTA_M, CTA_N, BLOCK_K, GROUP_SIZE_M,
    )
    cid = cluster.cluster_cta_id()

    # Cluster-aware SMEM layouts: A is M-split across CTAs, B is N-split.
    a_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [TILE_M, BLOCK_K], a_desc.dtype, cga_layout=a_cga_layout,
    )
    b_smem_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, TILE_N], b_desc.dtype, cga_layout=b_cga_layout,
    )
    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [STAGES, TILE_M, BLOCK_K], a_smem_layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [STAGES, BLOCK_K, TILE_N], b_smem_layout)

    # MMA arrival count for cluster mode with multicast=True must match
    # the HW arrival pattern -- ``tcgen05_mma_barrier_count(..., multicast=True)``
    # computes it from the operand layouts.
    mma_barrier_count: gl.constexpr = tcgen05_mma_barrier_count(
        [a_bufs.index(0), b_bufs.index(0)], multicast=True,
    )

    N_CONSUMERS: gl.constexpr = 3  # load, mma, epilogue
    load_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES, two_ctas=True)
    acc_empty_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES, two_ctas=True)
    acc_ready_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    for i in gl.static_range(STAGES):
        mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)
    for i in gl.static_range(ACC_STAGES):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    clc_barriers = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    clc_planar_ready_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES)
    clc_consumed_bars = mbarrier.allocate_mbarrier(batch=ACC_STAGES, two_ctas=True)
    for i in gl.static_range(ACC_STAGES):
        mbarrier.init(clc_barriers.index(i), count=1)
        mbarrier.init(clc_planar_ready_bars.index(i), count=1)
        mbarrier.init(clc_consumed_bars.index(i), count=N_CONSUMERS)

    cga_layout_clc: gl.constexpr = [[0]] * (gl.num_ctas().bit_length() - 1)
    clc_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0], cga_layout=cga_layout_clc)
    clc_result_buffers = gl.allocate_shared_memory(gl.int64, [ACC_STAGES, 2], clc_layout)
    clc_planar_pid_buffers = gl.allocate_shared_memory(gl.int64, [ACC_STAGES, 1], clc_layout)

    # TMEM accumulator ring: cluster-split along M (two_ctas=True); each CTA
    # physically holds its [CTA_M, TILE_N] slice.
    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(CTA_M, TILE_N),
        col_stride=1,
        cga_layout=cga_layout,
        two_ctas=True,
    )
    acc_bufs = allocate_tensor_memory(gl.float32, [ACC_STAGES, TILE_M, TILE_N], acc_layout)

    p = V4Args(
        config, cid,
        a_desc, b_desc, c_desc,
        a_bufs, b_bufs,
        acc_bufs,
        load_empty_bars, load_ready_bars,
        acc_empty_bars, acc_ready_bars,
        clc_result_buffers, clc_barriers, clc_planar_pid_buffers,
        clc_planar_ready_bars, clc_consumed_bars,
    )
    # Default partition (4 warps) = epilogue (needs >=4 warps for TMEM load).
    # Extra partitions (1 warp each) = mma, load, CLC scheduler.
    gl.warp_specialize(
        [
            (_v4_epilogue, (p,)),
            (_v4_mma, (p,)),
            (_v4_load, (p,)),
            (_v4_clc, (p,)),
        ],
        [1, 1, 1],
        [24, 24, 24],
    )


# ===-----------------------------------------------------------------------===#
# Host-side wrapper
# ===-----------------------------------------------------------------------===#


def _prepare_conv2d_inputs(input_tensor, weight_tensor, stride, padding, out=None):
    N, H, W, Ci = input_tensor.shape
    Co, R, S, Ci_w = weight_tensor.shape
    assert Ci == Ci_w, "Input and weight channel dimensions must match"
    if input_tensor.dtype != TORCH_GEMM_DTYPE or weight_tensor.dtype != TORCH_GEMM_DTYPE:
        raise ValueError(
            f"conv2d_im2col_2cta_ws_v4 expects bf16 input/weight, got "
            f"{input_tensor.dtype} and {weight_tensor.dtype}"
        )
    stride_h, stride_w = normalize_2d(stride, "stride")
    pad_h, pad_w = normalize_2d(padding, "padding")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"stride must be positive, got {(stride_h, stride_w)}")
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"padding must be non-negative, got {(pad_h, pad_w)}")

    input_tensor, weight_tensor = maybe_pad_ci_for_tma(input_tensor, weight_tensor)
    N, H, W, Ci = input_tensor.shape
    Co, R, S, _ = weight_tensor.shape

    out_h = (H + 2 * pad_h - R) // stride_h + 1
    out_w = (W + 2 * pad_w - S) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            "Invalid convolution geometry: computed output size "
            f"({out_h}, {out_w}) from H={H}, W={W}, R={R}, S={S}, "
            f"stride={(stride_h, stride_w)}, padding={(pad_h, pad_w)}."
        )

    output_shape = (N, out_h, out_w, Co)
    if out is None:
        output = torch.empty(output_shape, device=input_tensor.device, dtype=TORCH_GEMM_DTYPE)
    else:
        if out.shape != output_shape:
            raise ValueError(f"Output has invalid shape {out.shape}, expected {output_shape}")
        if out.device != input_tensor.device or out.dtype != TORCH_GEMM_DTYPE:
            raise ValueError("Output must match input device and bf16 dtype")
        if not out.is_contiguous():
            raise ValueError("Output must be contiguous NHWC")
        output = out
    return input_tensor, weight_tensor, output, N, H, W, Ci, Co, R, S, out_h, out_w, stride_h, stride_w, pad_h, pad_w


def _make_descriptors(
    input_tensor, weight_matrix, output_matrix,
    out_h, out_w, stride_h, stride_w, pad_h, pad_w,
    block_size_m, block_size_n, block_size_k, epilogue_block_n,
    cga_layout,
):
    tile_m = block_size_m * get_split_dim(cga_layout, 0)
    tile_n = block_size_n
    local_tile_n = tile_n // get_split_dim(_get_operand_cga_layout(cga_layout, 1), 1)

    a_block = [block_size_m, block_size_k]
    b_block = [block_size_k, local_tile_n]
    c_block = [tile_m, epilogue_block_n]

    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, GL_GEMM_DTYPE)
    b_layout = gl.NVMMASharedLayout.get_default_for(b_block, GL_GEMM_DTYPE)
    c_layout = gl.NVMMASharedLayout.get_default_for(c_block, GL_GEMM_DTYPE, cga_layout=cga_layout)

    _, H, W, _ = input_tensor.shape
    upper_h = (out_h - 1) * stride_h + 1 - H - pad_h
    upper_w = (out_w - 1) * stride_w + 1 - W - pad_w

    a_desc = TensorDescriptorIm2Col.from_tensor(
        input_tensor, a_block, a_layout, padding="zero",
        element_strides=[1, stride_h, stride_w, 1],
        pixel_box_lower_corner=[-pad_h, -pad_w],
        pixel_box_upper_corner=[upper_h, upper_w],
    )
    b_desc = TensorDescriptor.from_tensor(weight_matrix, b_block, b_layout)
    c_desc = TensorDescriptor.from_tensor(output_matrix, c_block, c_layout)
    return a_desc, b_desc, c_desc


def conv2d_im2col_2cta_ws_v4(
    input_tensor,
    weight_tensor,
    *,
    stride=1,
    padding=0,
    block_size_m=128,
    block_size_n=256,
    block_size_k=64,
    group_size_m=4,
    stages=5,
    acc_stages=2,
    epilogue_block_n=None,
    cga_layout=((1, 0),),
    num_warps=4,
    out=None,
):
    """Warp-specialized 2-CTA forward convolution (V4).

    Shape constraints:
        M_GEMM = N * out_h * out_w must be a multiple of ``block_size_m * 2``.
        ``block_size_n`` is the MMA tile width; ``epilogue_block_n`` controls
        how that tile is TMA-stored in smaller N-subtiles.

    Default config (1347.2 TFLOPS on target shape at Co=512):
        ``block_size_m=128, block_size_n=256, block_size_k=64, stages=5``.
    For Co tails with ``block_size_n=256`` (for example ``Co=384``), using
        ``epilogue_block_n=64`` lets the epilogue TMA-store four narrower
        subtiles without padding Co on the host.
    """
    if not is_blackwell():
        raise RuntimeError("This kernel requires a Blackwell CUDA GPU.")
    _validate_cga_layout(cga_layout)
    _set_default_tma_allocator(input_tensor.device)

    logical_Co = weight_tensor.shape[0]
    if epilogue_block_n is None:
        epilogue_block_n = 64 if logical_Co % block_size_n != 0 else block_size_n
    epilogue_block_n = int(epilogue_block_n)
    if epilogue_block_n <= 0 or epilogue_block_n > block_size_n or block_size_n % epilogue_block_n != 0:
        raise ValueError(
            f"epilogue_block_n={epilogue_block_n} must be a positive divisor of block_size_n={block_size_n}"
        )

    prep = _prepare_conv2d_inputs(input_tensor, weight_tensor, stride, padding, out=out)
    (input_tensor, weight_tensor, output, N, H, W, Ci, Co, R, S,
     out_h, out_w, stride_h, stride_w, pad_h, pad_w) = prep

    M_GEMM = N * out_h * out_w
    num_ctas = 2 ** len(cga_layout)
    tile_m = block_size_m * get_split_dim(cga_layout, 0)
    tile_n = block_size_n
    if M_GEMM % tile_m != 0:
        raise NotImplementedError(
            f"V4 requires M_GEMM to be a multiple of {tile_m}; got {M_GEMM}."
        )

    # B is stored as (K, N) for cluster MMA -- no in-kernel permute needed.
    weight_matrix = weight_tensor.reshape(Co, R * S * Ci).transpose(0, 1).contiguous()
    output_matrix = output.view(M_GEMM, Co)

    a_desc, b_desc, c_desc = _make_descriptors(
        input_tensor, weight_matrix, output_matrix,
        out_h, out_w, stride_h, stride_w, pad_h, pad_w,
        block_size_m, block_size_n, block_size_k, epilogue_block_n, cga_layout,
    )

    def grid(_meta):
        num_tiles = triton.cdiv(M_GEMM, tile_m) * triton.cdiv(Co, tile_n)
        return (num_tiles,)

    _conv2d_im2col_2cta_ws_v4_kernel[grid](
        a_desc, b_desc, c_desc,
        N, H, W, Ci, Co, R, S,
        out_h, out_w,
        stride_h, stride_w, pad_h, pad_w,
        GROUP_SIZE_M=group_size_m,
        STAGES=stages,
        ACC_STAGES=acc_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    return output


# ===-----------------------------------------------------------------------===#
# Correctness test
# ===-----------------------------------------------------------------------===#


@pytest.mark.parametrize("N,Ci,H,W,Co,R,S,stride,padding,block_size_n,epilogue_block_n,stages", [
    (128, 384, 64, 64, 384, 3, 3, 1, 1, 128, 128, 6),
    (128, 384, 64, 64, 384, 3, 3, 1, 1, 256, 64, 5),
    (128, 384, 64, 64, 512, 3, 3, 1, 1, 256, 256, 5),
])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell GPU (SM 10.x)")
def test_conv2d_im2col_2cta_ws_v4(N, Ci, H, W, Co, R, S, stride, padding, block_size_n, epilogue_block_n, stages):
    torch.manual_seed(0)
    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=TORCH_GEMM_DTYPE)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()

    triton_out = conv2d_im2col_2cta_ws_v4(
        x_nhwc, w_nhwc, stride=stride, padding=padding,
        block_size_n=block_size_n, epilogue_block_n=epilogue_block_n, stages=stages,
    )
    torch.cuda.synchronize()

    torch_out = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride, padding=padding)
    torch_out_nhwc = torch_out.permute(0, 2, 3, 1)
    torch.testing.assert_close(triton_out, torch_out_nhwc, atol=5e-2, rtol=5e-2)


# ===-----------------------------------------------------------------------===#
# Benchmarking
# ===-----------------------------------------------------------------------===#

BENCH_CASES = [
    {
        "N": 128,
        "Ci": 384,
        "H": 64,
        "W": 64,
        "Co": 384,
        "R": 3,
        "S": 3,
        "stride_val": 1,
        "pad_val": 1,
        "block_size_m": 128,
        "block_size_n": 128,
        "block_size_k": 128,
        "epilogue_block_n": 128,
        "stages": 4,
        "acc_stages": 2,
    },
    {
        "N": 128,
        "Ci": 384,
        "H": 64,
        "W": 64,
        "Co": 512,
        "R": 3,
        "S": 3,
        "stride_val": 1,
        "pad_val": 1,
        "block_size_m": 128,
        "block_size_n": 256,
        "block_size_k": 64,
        "epilogue_block_n": 256,
        "stages": 5,
        "acc_stages": 2,
    },
]


def _make_bench_inputs(N, H, W, Ci, Co, R, S):
    torch.manual_seed(0)

    x_nchw = torch.randn((N, Ci, H, W), device="cuda", dtype=TORCH_GEMM_DTYPE)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    w_nchw = torch.randn((Co, Ci, R, S), device="cuda", dtype=TORCH_GEMM_DTYPE)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()
    return x_nchw, x_nhwc, w_nchw, w_nhwc


def _benchmark_tflops(fn, *, N, H, W, Ci, Co, R, S, stride_val, pad_val):
    ms = triton.testing.do_bench(fn)
    out_h = (H + 2 * pad_val - R) // stride_val + 1
    out_w = (W + 2 * pad_val - S) // stride_val + 1
    flops = 2.0 * N * out_h * out_w * Co * Ci * R * S
    return flops * 1e-12 / (ms * 1e-3)


bench_configs = []
for case in BENCH_CASES:
    bench_configs.append(
        triton.testing.Benchmark(
            x_names=["kernel"],
            x_vals=["ws-2cta-v4"],
            line_arg="provider",
            line_vals=["gluon", "torch"],
            line_names=["Gluon (2-CTA WS)", "PyTorch"],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name=(
                "Conv2d2CTA "
                f"N={case['N']} Ci={case['Ci']} Co={case['Co']} "
                f"H={case['H']} W={case['W']} R={case['R']} S={case['S']} "
                f"stride={case['stride_val']} pad={case['pad_val']} "
                f"bm={case['block_size_m']} bn={case['block_size_n']} "
                f"bk={case['block_size_k']} epi={case['epilogue_block_n']} stages={case['stages']}"
            ),
            args=case,
        ))


@triton.testing.perf_report(bench_configs)
def bench(N, H, W, Ci, Co, R, S, stride_val, pad_val, block_size_m, block_size_n, block_size_k, epilogue_block_n,
          stages, acc_stages, kernel, provider):
    x_nchw, x_nhwc, w_nchw, w_nhwc = _make_bench_inputs(N, H, W, Ci, Co, R, S)

    if provider == "gluon":
        fn = lambda: conv2d_im2col_2cta_ws_v4(
            x_nhwc, w_nhwc,
            stride=stride_val, padding=pad_val,
            block_size_m=block_size_m, block_size_n=block_size_n,
            block_size_k=block_size_k, epilogue_block_n=epilogue_block_n, stages=stages,
            acc_stages=acc_stages,
        )
    elif provider == "torch":
        fn = lambda: torch.nn.functional.conv2d(x_nchw, w_nchw, stride=stride_val, padding=pad_val)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return _benchmark_tflops(
        fn,
        N=N,
        H=H,
        W=W,
        Ci=Ci,
        Co=Co,
        R=R,
        S=S,
        stride_val=stride_val,
        pad_val=pad_val,
    )


if __name__ == "__main__":
    bench.run(save_path=".", print_data=True)
