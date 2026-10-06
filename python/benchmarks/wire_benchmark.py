"""JSON vs axi-wire for E edges (Python/numpy reference; NOT the Rust implementation)."""
import json, sys, time, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from axi import wire


def best(f, reps=20):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts) * 1e6


def main(E=10_000, N=2_000):
    rng = np.random.default_rng(0)
    s, d = rng.integers(0, N, E), rng.integers(0, N, E); r = rng.integers(0, 60, E); w = rng.random(E).astype(np.float32)
    recs = [{"src": int(a), "dst": int(b), "rel": int(c), "w": float(x)} for a, b, c, x in zip(s, d, r, w)]
    js = json.dumps(recs); frame = wire.encode(0, 1, N, E, wire.pack_coo(s, d, r, w))
    # correctness first: both transports must carry the same graph
    back = wire.unpack_coo(wire.decode(frame)[1]); assert (back["src"] == s).all() and np.array_equal(back["weight"], w)
    rows = [("JSON dumps", best(lambda: json.dumps(recs))), ("JSON loads", best(lambda: json.loads(js))),
            ("wire pack_coo+encode (pure-py CRC32C)", best(lambda: wire.encode(0, 1, N, E, wire.pack_coo(s, d, r, w)), 3)),
            ("wire pack_coo only (no CRC)", best(lambda: wire.pack_coo(s, d, r, w))),
            ("wire decode, verify=False + unpack (zero-copy)", best(lambda: wire.unpack_coo(wire.decode(frame, verify=False)[1]))),
            ("wire decode, verify=True (pure-py CRC32C)", best(lambda: wire.decode(frame), 3))]
    print(f"E={E}  JSON={len(js)/1e3:.0f} KB  wire={len(frame)/1e3:.0f} KB")
    for k, v in rows: print(f"  {k:48s} {v:12.1f} us")
    print("NOTE: the pure-Python CRC32C is slow by construction; a hardware CRC32C (SSE4.2/ARM) runs at GB/s. Measure the Rust crate with cargo bench.")


if __name__ == "__main__": main()
