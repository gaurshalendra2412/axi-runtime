import json, random, zlib
import numpy as np
from axi import wire
from axi.wire import WireError


def test_crc32c_known_vectors_and_it_is_not_crc32_ieee():
    assert wire.crc32c(b"123456789") == 0xE3069283          # standard CRC-32C check value
    assert wire.crc32c(b"") == 0 and wire.crc32c(b"\x00" * 32) == 0x8A9136AA   # RFC 3720 test vector
    assert zlib.crc32(b"123456789") == 0xCBF43926 != wire.crc32c(b"123456789")   # what crc32fast would produce


def test_header_layout_offsets():
    f = wire.encode(0, 0x0102030405060708, 7, 9, b"abc")
    assert f[:4] == b"AXI\x01" and f[8:16] == bytes(range(1, 9)) and f[16:18] == b"\x00\x07" and f[20:24] == b"\x00\x00\x00\x09"
    assert f[24:32] == (3).to_bytes(8, "big") and len(f) == 32 + 8 + 4          # 3 payload bytes padded to 8


def test_coo_roundtrip_zero_copy_and_alignment():
    rng = np.random.default_rng(0); e = 1000
    s, d = rng.integers(0, 500, e), rng.integers(0, 500, e); r = rng.integers(0, 60, e); w = rng.random(e).astype(np.float32)
    frame = wire.encode(wire.MODE_COO, 42, 500, e, wire.pack_coo(s, d, r, w))
    h, p = wire.decode(frame); a = wire.unpack_coo(p)
    assert h["txid"] == 42 and h["edge_count"] == e
    assert (a["src"] == s).all() and (a["dst"] == d).all() and (a["rel"] == r).all() and np.array_equal(a["weight"], w)
    assert np.shares_memory(a, np.frombuffer(frame, np.uint8))              # truly a view, no copy
    assert (wire.HEADER_SIZE % 8) == 0


def test_bitmatrix_roundtrip_various_sizes():
    rng = np.random.default_rng(1)
    for n in (1, 7, 63, 64, 65, 130):
        adj = rng.random((n, n)) < .3
        assert np.array_equal(wire.unpack_bitmatrix(wire.pack_bitmatrix(adj), n), adj), n


def test_every_single_bit_flip_is_detected():
    frame = bytearray(wire.encode(0, 1, 3, 2, wire.pack_coo([0, 1], [1, 2], [1, 1], [1.0, 2.0])))
    for byte in range(len(frame)):
        for bit in range(8):
            frame[byte] ^= 1 << bit
            try: wire.decode(bytes(frame)); raise AssertionError(f"flip {byte}:{bit} undetected")
            except WireError: pass
            frame[byte] ^= 1 << bit


def test_truncation_and_hostile_length_fields_rejected():
    f = wire.encode(0, 1, 3, 1, b"x" * 40)
    for cut in range(len(f)):
        try: wire.decode(f[:cut]); raise AssertionError(cut)
        except WireError: pass
    bad = bytearray(f); bad[24:32] = (2 ** 63).to_bytes(8, "big")            # absurd payload_length
    try: wire.decode(bytes(bad)); raise AssertionError("accepted")
    except WireError: pass


def test_random_garbage_never_decodes_or_crashes():
    rng = random.Random(0)
    for _ in range(3000):
        b = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 80)))
        try: wire.decode(b); raise AssertionError("garbage decoded")
        except WireError: pass


def test_limits():
    try: wire.encode(0, 1, 70000, 0, b""); raise AssertionError
    except WireError: pass


def test_density_threshold_is_1_over_128_not_0_65():
    n = 1024
    for e in (int(n * n * x) for x in (0.001, 0.005, 0.0078, 0.0079, 0.01, 0.1, 0.65)):
        s = wire.payload_sizes(n, e)
        assert (wire.choose_mode(n, e) == wire.MODE_COO) == (s["coo"] <= s["bitmatrix"])
    assert wire.choose_mode(n, int(n * n * 0.0078)) == wire.MODE_COO and wire.choose_mode(n, int(n * n * 0.0079)) == wire.MODE_BITMATRIX
    s = wire.payload_sizes(n, int(n * n * 0.65)); assert s["coo"] > 80 * s["bitmatrix"]      # the pasted rule wastes >80x here


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
