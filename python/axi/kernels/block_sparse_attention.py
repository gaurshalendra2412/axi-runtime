"""
Block-sparse attention (forward + backward) over an indirection table of
ACTIVE KV blocks.

K/V live in a block pool [Total_Blocks, BLOCK_N, H, D]. For each sequence a
block table lists the active physical blocks, and block_lens gives how many
tokens in each active block are valid (0 < len <= BLOCK_N). Evicting a block =
removing it from the table; the pool is never copied or shifted.

Semantics (important):
  * Every query attends to ALL valid tokens in the sequence's active blocks.
  * There is NO causal mask. Intended for decode-style steps. Do not use for
    causal prefill without adding a mask.
  * Dropping blocks changes the model's output vs. full attention; the quality
    cost must be measured separately.
  * TRAINING: each physical block may appear at most once across the whole
    batch's active tables. The dK/dV kernel gives every active block to its own
    program, so a duplicated block would be written twice (a race).
  * The gradient buffers dK/dV are pool-sized (zeros except active blocks), so
    the backward pass does NOT save memory vs. the pool itself.

This module imports without Triton/CUDA. Kernels exist only if Triton is
installed, and only run on CUDA tensors.
"""

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from triton.runtime.errors import OutOfResources

    HAS_TRITON = True
except ImportError:  # CPU-only environments (e.g. CI)
    HAS_TRITON = False

BLOCK_M = 64
BLOCK_N = 64
# Backward-pass launch configs (query-tile, key-sub-tile, pipeline stages),
# tried in order. Each physical KV block (BLOCK_N tokens) is processed in
# BLOCK_N // key-sub-tile pieces, so small-shared-memory GPUs (e.g. T4: 64 KB)
# can still run head_dim=128.
_BWD_CONFIGS = [(64, 64, 2), (32, 64, 1), (32, 32, 1), (16, 32, 1), (16, 16, 1)]


# --------------------------------------------------------------------------
# Reference (plain PyTorch, differentiable, runs anywhere)
# --------------------------------------------------------------------------
def reference_block_sparse_attention(
    q, k_pool, v_pool, block_table, block_lens, num_active, sm_scale=None
):
    """Ground truth in fp32 math. q: [B,H,M,D]; pools: [Total,BLOCK_N,H,D];
    block_table/block_lens: [B,Max_Active]; num_active: [B].
    A sequence with zero valid tokens yields all-zero output."""
    d = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    out = torch.zeros_like(q)
    for b in range(q.shape[0]):
        ks, vs = [], []
        for i in range(int(num_active[b])):
            blk = int(block_table[b, i])
            ln = int(block_lens[b, i])
            if ln > 0:
                ks.append(k_pool[blk, :ln])
                vs.append(v_pool[blk, :ln])
        if not ks:
            continue
        k = torch.cat(ks, dim=0).float()
        v = torch.cat(vs, dim=0).float()
        scores = torch.einsum("hmd,lhd->hml", q[b].float(), k) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hml,lhd->hmd", probs, v).to(q.dtype)
    return out


def assert_unique_blocks(block_table: torch.Tensor, num_active: torch.Tensor):
    """Raise if any physical block appears twice among the ACTIVE entries."""
    idx = torch.arange(block_table.shape[1], device=block_table.device)[None, :]
    valid = idx < num_active.to(block_table.device)[:, None]
    blocks = block_table[valid]
    if blocks.numel() != torch.unique(blocks).numel():
        raise ValueError(
            "A physical block appears more than once in the active block "
            "tables; the backward pass would write it from two programs."
        )


if HAS_TRITON:

    # ----------------------------------------------------------------------
    # Forward (also writes LSE = m + log(l) for the backward pass)
    # ----------------------------------------------------------------------
    @triton.jit
    def _fwd_kernel(
        Q, K_pool, V_pool, Block_Table, Block_Lens, Num_Active, Out, LSE,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kn, stride_kh, stride_kd,
        stride_vb, stride_vn, stride_vh, stride_vd,
        stride_tb, stride_tn,
        stride_ob, stride_oh, stride_om, stride_od,
        H, M,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1).to(tl.int64)
        off_z = tl.program_id(2).to(tl.int64)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        offs_n = tl.arange(0, BLOCK_N)
        mask_m = offs_m < M

        q_ptrs = (
            Q + off_z * stride_qb + off_h * stride_qh
            + offs_m64[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        n_active = tl.load(Num_Active + off_z)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for idx in range(0, n_active):
            phys = tl.load(Block_Table + off_z * stride_tb + idx * stride_tn).to(tl.int64)
            blen = tl.load(Block_Lens + off_z * stride_tb + idx * stride_tn)
            mask_n = offs_n < blen

            k_ptrs = (
                K_pool + phys * stride_kb + off_h * stride_kh
                + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
            )
            v_ptrs = (
                V_pool + phys * stride_vb + off_h * stride_vh
                + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            )
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)  # [D, N]
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # [N, D]

            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(qk - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_safe[:, None]
        lse = tl.where(l_i == 0.0, 0.0, m_i + tl.log(l_safe))

        out_ptrs = (
            Out + off_z * stride_ob + off_h * stride_oh
            + offs_m64[:, None] * stride_om + offs_d[None, :] * stride_od
        )
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])
        tl.store(LSE + (off_z * H + off_h) * M + offs_m64, lse, mask=mask_m)

    # ----------------------------------------------------------------------
    # Backward dQ: parallel over query tiles, loops over active KV blocks
    # (each block processed in NUM_SUB pieces of BLOCK_N tokens)
    # ----------------------------------------------------------------------
    @triton.jit
    def _bwd_dq_kernel(
        Q, dO, LSE, Delta, dQ, K_pool, V_pool,
        Block_Table, Block_Lens, Num_Active,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_dob, stride_doh, stride_dom, stride_dod,
        stride_dqb, stride_dqh, stride_dqm, stride_dqd,
        stride_kb, stride_kn, stride_kh, stride_kd,
        stride_vb, stride_vn, stride_vh, stride_vd,
        stride_tb, stride_tn,
        H, M,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_SUB: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_h = tl.program_id(1).to(tl.int64)
        off_z = tl.program_id(2).to(tl.int64)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m64 = offs_m.to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < M

        q = tl.load(
            Q + off_z * stride_qb + off_h * stride_qh
            + offs_m64[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None], other=0.0)
        do = tl.load(
            dO + off_z * stride_dob + off_h * stride_doh
            + offs_m64[:, None] * stride_dom + offs_d[None, :] * stride_dod,
            mask=mask_m[:, None], other=0.0)
        row = (off_z * H + off_h) * M + offs_m64
        lse_i = tl.load(LSE + row, mask=mask_m, other=0.0)
        delta_i = tl.load(Delta + row, mask=mask_m, other=0.0)

        dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        n_active = tl.load(Num_Active + off_z)

        for idx in range(0, n_active):
            phys = tl.load(Block_Table + off_z * stride_tb + idx * stride_tn).to(tl.int64)
            blen = tl.load(Block_Lens + off_z * stride_tb + idx * stride_tn)
            for sub in range(0, NUM_SUB):
                offs_n = sub * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < blen

                k = tl.load(
                    K_pool + phys * stride_kb + off_h * stride_kh
                    + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=mask_n[:, None], other=0.0)  # [N, D]
                v = tl.load(
                    V_pool + phys * stride_vb + off_h * stride_vh
                    + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=mask_n[:, None], other=0.0)  # [N, D]

                qk = tl.dot(q, tl.trans(k)) * sm_scale
                p = tl.exp(qk - lse_i[:, None])
                p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta_i[:, None]) * sm_scale
                dq_acc += tl.dot(ds.to(k.dtype), k)

        tl.store(
            dQ + off_z * stride_dqb + off_h * stride_dqh
            + offs_m64[:, None] * stride_dqm + offs_d[None, :] * stride_dqd,
            dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])

    # ----------------------------------------------------------------------
    # Backward dK/dV: one program per (ACTIVE physical block, sub-piece).
    # No atomics; requires every physical block to appear at most once.
    # ----------------------------------------------------------------------
    @triton.jit
    def _bwd_dkv_kernel(
        Q, dO, LSE, Delta, dK_pool, dV_pool, K_pool, V_pool,
        Block_Table, Block_Lens, Num_Active,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_dob, stride_doh, stride_dom, stride_dod,
        stride_kb, stride_kn, stride_kh, stride_kd,
        stride_vb, stride_vn, stride_vh, stride_vd,
        stride_dkb, stride_dkn, stride_dkh, stride_dkd,
        stride_dvb, stride_dvn, stride_dvh, stride_dvd,
        stride_tb, stride_tn,
        H, M,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_SUB: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        active_idx = pid0 // NUM_SUB
        sub = pid0 % NUM_SUB
        off_h = tl.program_id(1).to(tl.int64)
        off_z = tl.program_id(2).to(tl.int64)

        n_active = tl.load(Num_Active + off_z)
        if active_idx < n_active:
            phys = tl.load(Block_Table + off_z * stride_tb + active_idx * stride_tn).to(tl.int64)
            blen = tl.load(Block_Lens + off_z * stride_tb + active_idx * stride_tn)

            offs_n = sub * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, BLOCK_D)
            offs_m = tl.arange(0, BLOCK_M)
            mask_n = offs_n < blen

            k = tl.load(
                K_pool + phys * stride_kb + off_h * stride_kh
                + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0)  # [N, D]
            v = tl.load(
                V_pool + phys * stride_vb + off_h * stride_vh
                + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None], other=0.0)  # [N, D]

            dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
            dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

            num_m_tiles = (M + BLOCK_M - 1) // BLOCK_M
            for start_m in range(0, num_m_tiles):
                cur_m = start_m * BLOCK_M + offs_m
                cur_m64 = cur_m.to(tl.int64)
                mask_m = cur_m < M

                q = tl.load(
                    Q + off_z * stride_qb + off_h * stride_qh
                    + cur_m64[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None], other=0.0)
                do = tl.load(
                    dO + off_z * stride_dob + off_h * stride_doh
                    + cur_m64[:, None] * stride_dom + offs_d[None, :] * stride_dod,
                    mask=mask_m[:, None], other=0.0)
                row = (off_z * H + off_h) * M + cur_m64
                lse_i = tl.load(LSE + row, mask=mask_m, other=0.0)
                delta_i = tl.load(Delta + row, mask=mask_m, other=0.0)

                qk = tl.dot(q, tl.trans(k)) * sm_scale
                p = tl.exp(qk - lse_i[:, None])
                p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

                dv_acc += tl.dot(tl.trans(p).to(do.dtype), do)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta_i[:, None]) * sm_scale
                dk_acc += tl.dot(tl.trans(ds).to(q.dtype), q)

            tl.store(
                dK_pool + phys * stride_dkb + off_h * stride_dkh
                + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
                dk_acc.to(dK_pool.dtype.element_ty), mask=mask_n[:, None])
            tl.store(
                dV_pool + phys * stride_dvb + off_h * stride_dvh
                + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
                dv_acc.to(dV_pool.dtype.element_ty), mask=mask_n[:, None])

# --------------------------------------------------------------------------
# Python wrappers
# --------------------------------------------------------------------------
def _prepare(q, k_pool, v_pool, block_table, block_lens, num_active, sm_scale):
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if not (q.is_cuda and k_pool.is_cuda and v_pool.is_cuda):
        raise ValueError("requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be float16 or bfloat16")
    b, h, m, d = q.shape
    if d not in (16, 32, 64, 128):
        raise ValueError("head_dim must be one of 16, 32, 64, 128")
    if k_pool.shape[1] != BLOCK_N or tuple(k_pool.shape[2:]) != (h, d):
        raise ValueError(f"k_pool must be [Total_Blocks, {BLOCK_N}, {h}, {d}]")
    if v_pool.shape != k_pool.shape:
        raise ValueError("v_pool must have the same shape as k_pool")
    bt = block_table.to(device=q.device, dtype=torch.int32).contiguous()
    bl = block_lens.to(device=q.device, dtype=torch.int32).contiguous()
    na = num_active.to(device=q.device, dtype=torch.int32).contiguous()
    if bt.shape != bl.shape:
        raise ValueError("block_table and block_lens must have the same shape")
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d)
    return bt, bl, na, scale


def _forward(q, k_pool, v_pool, bt, bl, na, scale):
    b, h, m, d = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((b, h, m), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(m, BLOCK_M), h, b)
    _fwd_kernel[grid](
        q, k_pool, v_pool, bt, bl, na, out, lse, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        bt.stride(0), bt.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        h, m,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=d,
        num_warps=4 if d <= 64 else 8, num_stages=2,
    )
    return out, lse


def block_sparse_attention(
    q, k_pool, v_pool, block_table, block_lens, num_active, sm_scale=None
):
    """Inference forward pass (no gradients). Requires CUDA + Triton."""
    bt, bl, na, scale = _prepare(
        q, k_pool, v_pool, block_table, block_lens, num_active, sm_scale)
    out, _ = _forward(q, k_pool, v_pool, bt, bl, na, scale)
    return out


class _BlockSparseAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_pool, v_pool, bt, bl, na, scale):
        out, lse = _forward(q, k_pool, v_pool, bt, bl, na, scale)
        ctx.save_for_backward(q, k_pool, v_pool, out, lse, bt, bl, na)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k_pool, v_pool, out, lse, bt, bl, na = ctx.saved_tensors
        b, h, m, d = q.shape
        do = do.contiguous()
        delta = (do.float() * out.float()).sum(dim=-1).contiguous()  # [B,H,M]

        dq = torch.empty_like(q)
        dk_pool = torch.zeros_like(k_pool)
        dv_pool = torch.zeros_like(v_pool)
        nw = 4 if d <= 64 else 8

        for bm, bn, stages in _BWD_CONFIGS:
            try:
                _bwd_dq_kernel[(triton.cdiv(m, bm), h, b)](
                    q, do, lse, delta, dq, k_pool, v_pool, bt, bl, na, ctx.scale,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
                    v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
                    bt.stride(0), bt.stride(1),
                    h, m,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=d, NUM_SUB=BLOCK_N // bn,
                    num_warps=nw, num_stages=stages,
                )
                _bwd_dkv_kernel[(bt.shape[1] * (BLOCK_N // bn), h, b)](
                    q, do, lse, delta, dk_pool, dv_pool, k_pool, v_pool, bt, bl, na, ctx.scale,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
                    v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
                    dk_pool.stride(0), dk_pool.stride(1), dk_pool.stride(2), dk_pool.stride(3),
                    dv_pool.stride(0), dv_pool.stride(1), dv_pool.stride(2), dv_pool.stride(3),
                    bt.stride(0), bt.stride(1),
                    h, m,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=d, NUM_SUB=BLOCK_N // bn,
                    num_warps=nw, num_stages=stages,
                )
                break
            except OutOfResources:
                # Launch failed before running (not enough shared memory):
                # nothing was written, so retrying with a smaller config is safe.
                continue
        else:
            raise RuntimeError(
                "Backward kernels do not fit in this GPU's shared memory "
                "even with the smallest tile configuration."
            )
        return dq, dk_pool, dv_pool, None, None, None, None


def block_sparse_attention_train(
    q, k_pool, v_pool, block_table, block_lens, num_active,
    sm_scale=None, check_unique=True,
):
    """Differentiable forward (supports .backward()). Gradients flow to
    q, k_pool and v_pool. check_unique=True raises if a physical block is
    listed twice (costs a GPU sync)."""
    bt, bl, na, scale = _prepare(
        q, k_pool, v_pool, block_table, block_lens, num_active, sm_scale)
    if check_unique:
        assert_unique_blocks(bt, na)
    return _BlockSparseAttentionFn.apply(q, k_pool, v_pool, bt, bl, na, scale)
