"""
High-Throughput In-Place KV-Cache Patching Kernel.
Updates physical VRAM block slices directly without prefill recomputation.

Uses a Triton kernel when Triton is installed and the tensors live on CUDA;
otherwise falls back to a deterministic PyTorch loop.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def triton_patch_kv_slice_kernel(
        kv_cache_ptr,
        update_ptr,
        page_indices_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        stride_kv_page,
        stride_kv_head,
        stride_kv_dim,
        stride_up_idx,
        stride_up_head,
        stride_up_dim,
    ):
        pid_item = tl.program_id(0)
        pid_head = tl.program_id(1)

        target_page = tl.load(page_indices_ptr + pid_item)
        dim_offsets = tl.arange(0, head_dim)

        src_offset = (
            pid_item * stride_up_idx
            + pid_head * stride_up_head
            + dim_offsets * stride_up_dim
        )
        dst_offset = (
            target_page * stride_kv_page
            + pid_head * stride_kv_head
            + dim_offsets * stride_kv_dim
        )

        update_vals = tl.load(update_ptr + src_offset)
        tl.store(kv_cache_ptr + dst_offset, update_vals)

except ImportError:
    _HAS_TRITON = False


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _patch_fallback(kv_cache, updates, page_indices):
    for idx, page_id in enumerate(page_indices):
        kv_cache[page_id.item()] = updates[idx]


def launch_in_place_patch(
    kv_cache: torch.Tensor, updates: torch.Tensor, page_indices: torch.Tensor
):
    """Write updates[i] into kv_cache[page_indices[i]] in place."""
    num_updates, num_heads, head_dim = updates.shape
    use_triton = (
        _HAS_TRITON
        and kv_cache.is_cuda
        and updates.is_cuda
        and page_indices.is_cuda
        and _is_power_of_two(head_dim)  # tl.arange requires a power-of-two extent
    )
    if not use_triton:
        _patch_fallback(kv_cache, updates, page_indices)
        return

    grid = (num_updates, num_heads)
    triton_patch_kv_slice_kernel[grid](
        kv_cache,
        updates,
        page_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        stride_kv_page=kv_cache.stride(0),
        stride_kv_head=kv_cache.stride(1),
        stride_kv_dim=kv_cache.stride(2),
        stride_up_idx=updates.stride(0),
        stride_up_head=updates.stride(1),
        stride_up_dim=updates.stride(2),
    )
