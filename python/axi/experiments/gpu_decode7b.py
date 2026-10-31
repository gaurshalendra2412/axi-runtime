"""Test 32 (docs/GPU_DECODE_7B_TEST.md): tokens a second one NVIDIA GPU (an L4 in Colab) makes with a 7B-shape model, at AXI-sized and kept-chat-sized prompts. The PyTorch twin of tpu_decode.py (Test 31):
the exact SHAPE of Qwen2.5-7B (28 layers, hidden 3,584, 28 query heads, 4 key-value heads, head size 128, feed-forward 18,944, vocabulary 152,064), constant fp16 weights (a step takes the same time whatever the values),
a decode step for B sessions that reads each session's whole cache of L tokens (the new token's own key and value are used, not written back), timed with CUDA events, median of 15 after 3 warm-up steps."""
import json
import statistics

from axi.experiments.tpu_decode import AXI_TOKENS, FF, HD, HID, KEPT_MEAN, KVH, KV_BYTES_PER_TOKEN, LAYERS, QH, VOCAB, WEIGHT_PARAMS

PLAN = [("AXI step", AXI_TOKENS, [1, 16, 32]), ("kept chat at 50k", 50000, [1, 2]), ("kept chat at 100k", 100000, [1]),
        ("kept chat at 250k", 250000, [1]), ("kept chat, mean of the long run", KEPT_MEAN, [1])]
MARGIN = 1.2e9    # kept free for the step's own working memory


def cache_bytes(tokens, sessions):
    return tokens * sessions * KV_BYTES_PER_TOKEN


def measure(out_path="gpu_decode7b.json", say=print):
    import torch
    assert torch.cuda.is_available(), "no GPU"
    dev = "cuda"
    h16 = torch.float16

    def full(*shape):
        return torch.full(shape, 0.01, dtype=h16, device=dev)
    P = dict(embed=full(VOCAB, HID), head=full(HID, VOCAB), wq=full(LAYERS, HID, QH * HD), wk=full(LAYERS, HID, KVH * HD), wv=full(LAYERS, HID, KVH * HD),
             wo=full(LAYERS, QH * HD, HID), wg=full(LAYERS, HID, FF), wu=full(LAYERS, HID, FF), wd=full(LAYERS, FF, HID))
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, total_bytes=total, free_bytes_after_weights=free, weight_bytes=WEIGHT_PARAMS * 2, kv_bytes_per_token=KV_BYTES_PER_TOKEN, rows=[])
    say("weights on the GPU; free:", free / 1e9, "GB")

    def rms(x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h16)

    @torch.no_grad()
    def step(tokens, K, V):
        x = P["embed"][tokens]
        B = x.shape[0]
        for l in range(LAYERS):
            h = rms(x)
            q = (h @ P["wq"][l]).reshape(B, KVH, QH // KVH, HD)
            kn = (h @ P["wk"][l]).reshape(B, KVH, 1, HD)
            vn = (h @ P["wv"][l]).reshape(B, KVH, 1, HD)
            sc = torch.matmul(q, K[l].transpose(-1, -2)).float()                 # (B, KVH, 7, L): reads the whole key cache
            sn = (q * kn).sum(-1, keepdim=True).float()
            p = torch.softmax(torch.cat([sc, sn], -1) * (HD ** -0.5), -1).to(h16)
            o = torch.matmul(p[..., :-1], V[l]) + p[..., -1:] * vn               # reads the whole value cache
            x = x + o.reshape(B, QH * HD) @ P["wo"][l]
            h = rms(x)
            x = x + (torch.nn.functional.silu(h @ P["wg"][l]) * (h @ P["wu"][l])) @ P["wd"][l]
        return rms(x) @ P["head"]

    for name, L, bs in PLAN:
        for B in bs:
            need = cache_bytes(L, B)
            if need > free - MARGIN:
                row = dict(name=name, tokens=L, sessions=B, fits=False, need_cache_gb=need / 1e9)
                res["rows"].append(row); say(row); continue
            try:
                K = [torch.zeros((B, KVH, L, HD), dtype=h16, device=dev) for _ in range(LAYERS)]
                V = [torch.zeros((B, KVH, L, HD), dtype=h16, device=dev) for _ in range(LAYERS)]
                tok = torch.zeros((B,), dtype=torch.long, device=dev)
                for _ in range(3):
                    step(tok, K, V)
                torch.cuda.synchronize()
                ts = []
                for _ in range(15):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); step(tok, K, V); b.record(); torch.cuda.synchronize()
                    ts.append(a.elapsed_time(b))
                ms = statistics.median(ts)
                moved = WEIGHT_PARAMS * 2 + need
                row = dict(name=name, tokens=L, sessions=B, fits=True, step_ms=ms, tokens_per_second=B / (ms / 1000), gb_moved=moved / 1e9, gb_per_s=moved / 1e9 / (ms / 1000))
            except Exception as e:
                row = dict(name=name, tokens=L, sessions=B, fits=True, error=str(e)[:300])
            res["rows"].append(row); say({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
            K = V = None
            torch.cuda.empty_cache()
    json.dump(res, open(out_path, "w"), indent=1)
    return res
