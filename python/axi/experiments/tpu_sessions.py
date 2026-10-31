"""Test 30 (docs/TPU_SESSIONS_TEST.md): the sessions-per-chip count of Test 29, on a Google TPU (v6e-1 in Colab) with JAX.
Same question, same prompt lengths, same model shape (Qwen2.5-0.5B: 24 layers, 2 kv heads, head size 64, 16-bit). Real cache-shaped arrays are put on the chip, session after session, until it says out of memory.
Two things are recorded that the GPU run could not show: the chip's own memory limit, and the bytes a session really takes on the chip (TPUs pad array shapes to tile sizes, so a head size of 64 can cost more than 64)."""
import json

from axi.experiments.gpu_sessions import LENGTHS, SHAPES, bytes_per_token, count_until_full, fit_by_formula


def measure(out_path="tpu_sessions.json", say=print):
    import jax
    import jax.numpy as jnp
    dev = jax.devices()[0]
    assert dev.platform == "tpu", "this runtime has no TPU: " + str(dev)
    layers, kvh, hd = SHAPES["Qwen2.5-0.5B"]
    nominal = bytes_per_token(layers, kvh, hd)
    stats = dev.memory_stats()
    limit, used0 = stats["bytes_limit"], stats["bytes_in_use"]
    res = dict(device=str(dev.device_kind), jax=jax.__version__, shape=[layers, kvh, hd], nominal_bytes_per_token=nominal, bytes_limit=limit, bytes_in_use_at_start=used0)

    def used():
        return dev.memory_stats()["bytes_in_use"]

    def alloc(tokens):
        try:
            arrs = [jnp.zeros((2, 1, kvh, tokens, hd), dtype=jnp.bfloat16) for _ in range(layers)]
            for a in arrs:
                a.block_until_ready()
            return arrs
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                raise MemoryError from e
            raise

    def free_all(held):
        for h in held:
            for a in h:
                a.delete()
        held.clear()
    counts = {}
    for name, L in LENGTHS.items():
        before = used(); one = alloc(L); real = used() - before; free_all([one])
        n = count_until_full(alloc, free_all, L)
        counts[name] = dict(tokens=L, sessions_held=n, nominal_session_bytes=L * nominal, real_session_bytes=real, real_over_nominal=round(real / (L * nominal), 4),
                            by_nominal=fit_by_formula(limit - used0, L, nominal), by_real=int((limit - used0) // real) if real else None)
        say(name, counts[name])
    res["counts"] = counts
    json.dump(res, open(out_path, "w"), indent=1)
    return res
