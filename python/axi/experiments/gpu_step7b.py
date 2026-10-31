"""Test 33 (docs/GPU_STEP_7B_TEST.md): the whole step on an L4, with the 7B-shape model of Test 32. Test 32 timed decode only. This adds the part it left out: the PREFILL, the pass over the prompt tokens before the first reply token.
AXI sends a fresh prompt of 2,824 tokens every step, so every step starts with a prefill of them. A kept chat that reuses its cache has only its newest tokens (assumed 100: a user turn and a tool result) to prefill, but they attend to the whole cache.
Measured here: the time of a prefill of n tokens for n = 100 ... 32,768 (batch 1, no cache). Combined, in `combine`, with the decode times saved in Test 32."""
import json
import statistics

from axi.experiments.tpu_decode import FF, HD, HID, KVH, LAYERS, QH, VOCAB, WEIGHT_PARAMS

NS = [100, 512, 2824, 8192, 32768]
N_NEW = 100          # tokens a kept chat adds per step (assumption, stated on the page)
REPLY_TOKENS = [10, 40, 100]


def measure(out_path="gpu_step7b.json", say=print):
    import torch
    import torch.nn.functional as F
    h16 = torch.float16
    dev = "cuda"

    def full(*shape):
        return torch.full(shape, 0.01, dtype=h16, device=dev)
    P = dict(embed=full(VOCAB, HID), wq=full(LAYERS, HID, QH * HD), wk=full(LAYERS, HID, KVH * HD), wv=full(LAYERS, HID, KVH * HD),
             wo=full(LAYERS, QH * HD, HID), wg=full(LAYERS, HID, FF), wu=full(LAYERS, HID, FF), wd=full(LAYERS, FF, HID))

    def rms(x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h16)

    @torch.no_grad()
    def prefill(n):
        x = P["embed"][torch.zeros(n, dtype=torch.long, device=dev)]
        for l in range(LAYERS):
            h = rms(x)
            q = (h @ P["wq"][l]).reshape(1, n, QH, HD).transpose(1, 2)
            k = (h @ P["wk"][l]).reshape(1, n, KVH, HD).transpose(1, 2)
            v = (h @ P["wv"][l]).reshape(1, n, KVH, HD).transpose(1, 2)
            try:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            except TypeError:
                o = F.scaled_dot_product_attention(q, k.repeat_interleave(QH // KVH, 1), v.repeat_interleave(QH // KVH, 1), is_causal=True)
            x = x + o.transpose(1, 2).reshape(n, QH * HD) @ P["wo"][l]
            h = rms(x)
            x = x + (F.silu(h @ P["wg"][l]) * (h @ P["wu"][l])) @ P["wd"][l]
        return x
    res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rows=[])
    for n in NS:
        try:
            for _ in range(2):
                prefill(n)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); prefill(n); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
            ms = statistics.median(ts)
            row = dict(tokens=n, prefill_ms=ms, tokens_per_second=n / (ms / 1000), tflops=2 * WEIGHT_PARAMS * n / (ms / 1000) / 1e12)
        except Exception as e:
            row = dict(tokens=n, error=str(e)[:300])
            torch.cuda.empty_cache()
        res["rows"].append(row); say({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    json.dump(res, open(out_path, "w"), indent=1)
    return res


def interp(points, n):
    """Prefill ms at n tokens from the measured (tokens, ms) points, straight lines between them; None outside the measured range."""
    pts = sorted(points)
    if n < pts[0][0] or n > pts[-1][0]:
        return None
    for (a, ta), (b, tb) in zip(pts, pts[1:]):
        if a <= n <= b:
            return ta + (tb - ta) * (n - a) / (b - a)
    return pts[-1][1]


def combine(decode_rows, prefill_rows, R):
    """Chip seconds a session needs for one whole step (prefill, then R reply tokens), and exact steps per chip-second, from the saved timings.
    decode_rows: Test 32's rows; prefill_rows: this test's rows. A batch shares each decode step (step_ms / B per session); each session pays its own prefill (compute-bound, not shared).
    A kept chat's cached prefill is the prefill of N_NEW tokens plus one read of its cache (the extra decode time of its length over the AXI length at one session)."""
    dec = {(r["name"], r["sessions"]): r["step_ms"] for r in decode_rows if r.get("step_ms")}
    pts = [(r["tokens"], r["prefill_ms"]) for r in prefill_rows if r.get("prefill_ms")]
    p_axi, p_new = interp(pts, 2824), interp(pts, N_NEW)
    base = dec[("AXI step", 1)]
    out = {}
    for (name, B), step in dec.items():
        per_decode = R * step / B
        if name == "AXI step":
            per = p_axi + per_decode
        else:
            L = {"kept chat at 50k": 50000, "kept chat at 100k": 100000}[name]
            per = p_new + max(dec[(name, 1)] - base, 0.0) + per_decode
        out[(name, B)] = dict(chip_seconds_per_step=per / 1000, steps_per_chip_second=1000 / per)
    return out


def first_token(decode_rows, prefill_rows):
    """Time to the first reply token at one session (ms): AXI = prefill of 2,824 + one decode step; a cached kept chat = prefill of N_NEW + one cache read + one decode step;
    an uncached kept chat = the prefill of its whole prompt (only the largest measured prompt, 32,768 tokens, is timed; the real prompt is longer, so this is a lower bound)."""
    dec = {(r["name"], r["sessions"]): r["step_ms"] for r in decode_rows if r.get("step_ms")}
    pts = [(r["tokens"], r["prefill_ms"]) for r in prefill_rows if r.get("prefill_ms")]
    base = dec[("AXI step", 1)]
    res = {"AXI step": interp(pts, 2824) + base}
    for name in ("kept chat at 50k", "kept chat at 100k"):
        if (name, 1) not in dec:
            continue
        res[name + ", cache kept"] = interp(pts, N_NEW) + max(dec[(name, 1)] - base, 0.0) + base
    res["kept chat, cache lost (at least)"] = max(p for _, p in pts if p) + base if pts else None
    return res
