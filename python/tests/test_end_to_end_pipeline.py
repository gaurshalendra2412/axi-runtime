import torch
from axi.kernels.triton_slice_patch import launch_in_place_patch

def test_pipeline_transaction():
    num_pages, num_heads, head_dim = 4, 8, 64
    kv_cache = torch.zeros((num_pages, num_heads, head_dim), dtype=torch.float32)

    candidate_update = torch.ones((1, num_heads, head_dim), dtype=torch.float32)
    target_page = torch.tensor([2], dtype=torch.int32)

    launch_in_place_patch(kv_cache, candidate_update, target_page)

    assert torch.equal(kv_cache[2], candidate_update[0]), "Memory patch failed"
    assert torch.equal(kv_cache[0], torch.zeros_like(kv_cache[0])), "Unrelated page corrupted"
    print("AXI End-to-End Pipeline Slice Check: PASSED")

if __name__ == "__main__":
    test_pipeline_transaction()
