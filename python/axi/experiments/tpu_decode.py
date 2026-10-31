"""Test 31 (docs/TPU_DECODE_TEST.md): tokens a second one Google TPU chip makes with a 7B-shape model, at AXI-sized and kept-chat-sized prompts.
The model is the exact SHAPE of Qwen2.5-7B (28 layers, hidden 3,584, 28 query heads, 4 key-value heads, head size 128, feed-forward 18,944, vocabulary 152,064) with constant weights: how long a step takes does not depend on the values.
One jitted decode step for B sessions: embed, then per layer attention over a cache of L tokens per session (read-only; the new token's own key and value are included) and the feed-forward, then the output layer.
Weights and caches are real bf16 buffers on the chip; the step is timed after compilation, median of 15."""
import json
import statistics
import time

LAYERS, HID, QH, KVH, HD, FF, VOCAB = 28, 3584, 28, 4, 128, 18944, 152064
KV_BYTES_PER_TOKEN = 2 * LAYERS * KVH * HD * 2                 # 57,344
WEIGHT_PARAMS = LAYERS * (HID * QH * HD + 2 * HID * KVH * HD + QH * HD * HID + 3 * HID * FF) + 2 * VOCAB * HID
AXI_TOKENS, KEPT_MEAN = 2824, 476835
PLAN = [("AXI step", AXI_TOKENS, [1, 16, 64]), ("kept chat at 50k", 50000, [1, 2, 4]), ("kept chat at 100k", 100000, [1, 2]),
        ("kept chat at 250k", 250000, [1]), ("kept chat, mean of the long run", KEPT_MEAN, [1])]


def need_bytes(tokens, sessions):
    """Weights plus caches, in bytes (bf16)."""
    return WEIGHT_PARAMS * 2 + tokens * sessions * KV_BYTES_PER_TOKEN


def measure(out_path="tpu_decode.json", say=print):
    import jax
    import jax.numpy as jnp
    from jax import lax
    dev = jax.devices()[0]
    assert dev.platform == "tpu", "no TPU: " + str(dev)
    limit = dev.memory_stats()["bytes_limit"]
    bf = jnp.bfloat16
    P = dict(embed=jnp.full((VOCAB, HID), 0.01, bf), head=jnp.full((HID, VOCAB), 0.01, bf),
             wq=jnp.full((LAYERS, HID, QH * HD), 0.01, bf), wk=jnp.full((LAYERS, HID, KVH * HD), 0.01, bf), wv=jnp.full((LAYERS, HID, KVH * HD), 0.01, bf),
             wo=jnp.full((LAYERS, QH * HD, HID), 0.01, bf), wg=jnp.full((LAYERS, HID, FF), 0.01, bf), wu=jnp.full((LAYERS, HID, FF), 0.01, bf), wd=jnp.full((LAYERS, FF, HID), 0.01, bf))
    jax.block_until_ready(P)
    res = dict(device=str(dev.device_kind), jax=jax.__version__, bytes_limit=limit, weight_bytes=WEIGHT_PARAMS * 2, kv_bytes_per_token=KV_BYTES_PER_TOKEN, rows=[])

    def rms(x):
        x32 = x.astype(jnp.float32)
        return (x32 * lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + 1e-6)).astype(bf)

    @jax.jit
    def step(P, tokens, K, V):
        x = P["embed"][tokens]                                            # (B, HID)
        B = x.shape[0]

        def layer(x, xs):
            wq, wk, wv, wo, wg, wu, wd, k, v = xs                         # k, v: (B, KVH, L, HD)
            h = rms(x)
            q = (h @ wq).reshape(B, KVH, QH // KVH, HD)
            kn = (h @ wk).reshape(B, KVH, 1, HD); vn = (h @ wv).reshape(B, KVH, 1, HD)
            sc = jnp.einsum("bghd,bgld->bghl", q, k).astype(jnp.float32)
            sn = jnp.einsum("bghd,bgld->bghl", q, kn).astype(jnp.float32)
            s = jnp.concatenate([sc, sn], -1) * (HD ** -0.5)
            p = jax.nn.softmax(s, -1).astype(bf)
            o = jnp.einsum("bghl,bgld->bghd", p[..., :-1], v) + jnp.einsum("bghl,bgld->bghd", p[..., -1:], vn)
            x = x + o.reshape(B, QH * HD) @ wo
            h = rms(x)
            x = x + (jax.nn.silu(h @ wg) * (h @ wu)) @ wd
            return x, None
        x, _ = lax.scan(layer, x, (P["wq"], P["wk"], P["wv"], P["wo"], P["wg"], P["wu"], P["wd"], K, V))
        return rms(x) @ P["head"]

    for name, L, bs in PLAN:
        for B in bs:
            if need_bytes(L, B) > limit - (1 << 30):
                row = dict(name=name, tokens=L, sessions=B, fits=False, need_gb=need_bytes(L, B) / 1e9)
                res["rows"].append(row); say(row); continue
            try:
                K = jnp.zeros((LAYERS, B, KVH, L, HD), bf); V = jnp.zeros((LAYERS, B, KVH, L, HD), bf)
                tok = jnp.zeros((B,), jnp.int32)
                jax.block_until_ready((K, V))
                t0 = time.time(); step(P, tok, K, V).block_until_ready(); compile_s = time.time() - t0
                for _ in range(3):
                    step(P, tok, K, V).block_until_ready()
                ts = []
                for _ in range(15):
                    a = time.perf_counter(); step(P, tok, K, V).block_until_ready(); ts.append((time.perf_counter() - a) * 1000)
                ms = statistics.median(ts)
                moved = WEIGHT_PARAMS * 2 + B * L * KV_BYTES_PER_TOKEN
                row = dict(name=name, tokens=L, sessions=B, fits=True, step_ms=ms, tokens_per_second=B / (ms / 1000), gb_moved=moved / 1e9, gb_per_s=moved / 1e9 / (ms / 1000), compile_s=compile_s)
            except Exception as e:
                row = dict(name=name, tokens=L, sessions=B, fits=True, error=str(e)[:300])
            res["rows"].append(row); say({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
            del K, V
    json.dump(res, open(out_path, "w"), indent=1)
    return res
