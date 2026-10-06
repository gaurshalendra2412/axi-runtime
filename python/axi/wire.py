"""RFC-001 axi-wire reference codec (Python). Spec decisions made explicit where the draft was ambiguous:

  header 32 bytes, all multibyte fields BIG-endian:
    0  magic 'AXI\\x01' | 4 mode u8 | 5 compression u8 | 6 schema u8 | 7 reserved=0
    8  txid u64 | 16 node_count u16 | 18 reserved u16=0 | 20 edge_count u32 | 24 payload_length u64
  payload: payload_length bytes, then zero padding to a multiple of 8
  trailer: CRC-32C (Castagnoli, poly 0x1EDC6F41 reflected 0x82F63B78) over header+payload+padding, u32 BE
  COO tuple (16 B): src u32 | dst u32 | rel u16 | flags u16 | weight f32   (all BE)
  Bitmatrix: n rows x ceil(n/64) u64 BE words; bit j of row i is bit (j % 64) of word j//64 (LSB first)
NOTE: crc32fast / zlib.crc32 are CRC-32 (IEEE), a DIFFERENT checksum. This module implements CRC-32C.
"""
import struct
import numpy as np

MAGIC = b"AXI\x01"
HEADER = struct.Struct(">4sBBBBQHHIQ")
HEADER_SIZE = 32
assert HEADER.size == HEADER_SIZE
MODE_COO, MODE_BITMATRIX, MODE_DIAG, MODE_CISC = 0, 1, 2, 3
COO_DTYPE = np.dtype([("src", ">u4"), ("dst", ">u4"), ("rel", ">u2"), ("flags", ">u2"), ("weight", ">f4")])
assert COO_DTYPE.itemsize == 16


class WireError(ValueError):
    pass


def _crc_table():
    t = []
    for n in range(256):
        c = n
        for _ in range(8): c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        t.append(c)
    return t


_T = _crc_table()


def crc32c(data) -> int:
    c = 0xFFFFFFFF
    for b in bytes(data): c = _T[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _pad8(n): return (n + 7) & ~7


def encode(mode, txid, node_count, edge_count, payload, compression=0, schema=1) -> bytes:
    if node_count > 0xFFFF: raise WireError("node_count exceeds u16 (65535)")
    if mode not in (0, 1, 2, 3): raise WireError("unknown mode")
    if compression != 0: raise WireError("compression not implemented")
    payload = bytes(payload)
    body = HEADER.pack(MAGIC, mode, compression, schema, 0, txid, node_count, 0, edge_count, len(payload)) + payload
    body += b"\x00" * (_pad8(len(payload)) - len(payload))
    return body + struct.pack(">I", crc32c(body))


def decode(buf, verify=True):
    """-> (header dict, payload memoryview). Payload is a zero-copy view into buf."""
    mv = memoryview(buf)
    if len(mv) < HEADER_SIZE + 4: raise WireError("truncated frame")
    magic, mode, comp, schema, rsv, txid, nodes, rsv2, edges, plen = HEADER.unpack_from(mv, 0)
    if magic != MAGIC: raise WireError("bad magic")
    if schema != 1: raise WireError("unsupported schema version")
    if mode not in (0, 1, 2, 3): raise WireError("unknown mode")
    if comp != 0: raise WireError("unsupported compression")
    if rsv != 0 or rsv2 != 0: raise WireError("reserved fields must be zero")
    total = HEADER_SIZE + _pad8(plen) + 4
    if plen > len(mv) or len(mv) < total: raise WireError("payload truncated")   # checked before any arithmetic on plen is trusted
    if verify and crc32c(mv[:total - 4]) != struct.unpack_from(">I", mv, total - 4)[0]: raise WireError("CRC mismatch")
    if any(mv[HEADER_SIZE + plen:total - 4]): raise WireError("nonzero padding")
    return dict(mode=mode, txid=txid, node_count=nodes, edge_count=edges, payload_length=plen), mv[HEADER_SIZE:HEADER_SIZE + plen]


def pack_coo(src, dst, rel, weight, flags=None) -> bytes:
    a = np.zeros(len(src), COO_DTYPE); a["src"], a["dst"], a["rel"], a["weight"] = src, dst, rel, weight
    if flags is not None: a["flags"] = flags
    return a.tobytes()


def unpack_coo(payload) -> np.ndarray:
    if len(payload) % 16: raise WireError("COO payload not a multiple of 16")
    return np.frombuffer(payload, COO_DTYPE)             # zero-copy view (big-endian dtype)


def pack_bitmatrix(adj: np.ndarray) -> bytes:
    n = adj.shape[0]; words = (n + 63) // 64
    packed = np.packbits(adj.astype(bool), axis=1, bitorder="little")           # n x ceil(n/8)
    padded = np.zeros((n, words * 8), np.uint8); padded[:, :packed.shape[1]] = packed
    return padded.view("<u8").astype(">u8").tobytes()


def unpack_bitmatrix(payload, n: int) -> np.ndarray:
    words = (n + 63) // 64
    if len(payload) != n * words * 8: raise WireError("bitmatrix size mismatch")
    w = np.frombuffer(payload, ">u8").reshape(n, words).astype("<u8")
    return np.unpackbits(w.view(np.uint8), axis=1, bitorder="little")[:, :n].astype(bool)


def payload_sizes(n_nodes, n_edges):
    return {"coo": 16 * n_edges, "bitmatrix": n_nodes * ((n_nodes + 63) // 64) * 8}


def choose_mode(n_nodes, n_edges) -> int:
    s = payload_sizes(n_nodes, n_edges)
    return MODE_COO if s["coo"] <= s["bitmatrix"] else MODE_BITMATRIX
