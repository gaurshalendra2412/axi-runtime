"""Grammar-constrained decoding for the CICO delta language.

A byte-level DFA recognises exactly the strings CICOParser accepts. TokenMasker lifts it to
arbitrary (BPE-style) token vocabularies: a token is allowed in DFA state q iff feeding all of
its bytes never hits the reject sink. Masks are computed by walking a vocabulary byte-trie in
lockstep with the DFA and cached per DFA state.
"""
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

DIG = b"0123456789"
REL = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
SEPS = b" \n"


class CICOGrammarDFA:
    START = "START"

    def __init__(self):
        T: Dict[str, Dict[int, str]] = {}

        def e(s, chars, t):
            for ch in chars:
                T.setdefault(s, {})[ch] = t
            T.setdefault(t, {})

        S = self.START
        e(S, b"A", "A1"); e(S, b"D", "D1")
        e("A1", b"D", "A2"); e("A2", b"D", "A3"); e("A3", b"[", "AO")
        e("D1", b"E", "D2"); e("D2", b"L", "D3"); e("D3", b"[", "DO")
        # node add: ADD[r,c:-?v]
        e("AO", DIG, "NA_r"); e("NA_r", DIG, "NA_r"); e("NA_r", b",", "NA_c0")
        e("NA_c0", DIG, "NA_c"); e("NA_c", DIG, "NA_c"); e("NA_c", b":", "NA_v0")
        e("NA_v0", b"-", "NA_vs"); e("NA_v0", DIG, "NA_v"); e("NA_vs", DIG, "NA_v")
        e("NA_v", DIG, "NA_v"); e("NA_v", b"]", "END")
        # node del: DEL[r,c]
        e("DO", DIG, "ND_r"); e("ND_r", DIG, "ND_r"); e("ND_r", b",", "ND_c0")
        e("ND_c0", DIG, "ND_c"); e("ND_c", DIG, "ND_c"); e("ND_c", b"]", "END")
        # edge: [(r,c)->(r,c):rel#-?w]   (shared by ADD and DEL after the '[')
        e("AO", b"(", "E_u0"); e("DO", b"(", "E_u0")
        e("E_u0", DIG, "E_ur"); e("E_ur", DIG, "E_ur"); e("E_ur", b",", "E_uc0")
        e("E_uc0", DIG, "E_uc"); e("E_uc", DIG, "E_uc"); e("E_uc", b")", "E_a1")
        e("E_a1", b"-", "E_a2"); e("E_a2", b">", "E_a3"); e("E_a3", b"(", "E_v0")
        e("E_v0", DIG, "E_vr"); e("E_vr", DIG, "E_vr"); e("E_vr", b",", "E_vc0")
        e("E_vc0", DIG, "E_vc"); e("E_vc", DIG, "E_vc"); e("E_vc", b")", "E_c")
        e("E_c", b":", "E_rel0")
        e("E_rel0", REL, "E_rel"); e("E_rel", REL, "E_rel"); e("E_rel", b"#", "E_w0")
        e("E_w0", b"-", "E_ws"); e("E_w0", DIG, "E_w"); e("E_ws", DIG, "E_w")
        e("E_w", DIG, "E_w"); e("E_w", b"]", "END")
        # separators
        e("END", SEPS, "SEP"); e("SEP", b"A", "A1"); e("SEP", b"D", "D1")
        self.T = T
        self.accepting = {S, "END", "SEP"}
        self._dist = self._distances()

    def step(self, state: Optional[str], byte: int) -> Optional[str]:
        return None if state is None else self.T[state].get(byte)

    def run(self, data: bytes, state: Optional[str] = START) -> Optional[str]:
        for b in data:
            state = self.step(state, b)
            if state is None:
                return None
        return state

    def is_accepting(self, state: Optional[str]) -> bool:
        return state in self.accepting

    def accepts(self, data: bytes) -> bool:
        return self.is_accepting(self.run(data))

    def states(self) -> List[str]:
        seen, q = {self.START}, deque([self.START])
        while q:
            s = q.popleft()
            for t in self.T[s].values():
                if t not in seen:
                    seen.add(t); q.append(t)
        return sorted(seen)

    def _distances(self) -> Dict[str, int]:
        rev: Dict[str, List[str]] = {}
        for s, tr in self.T.items():
            for t in tr.values():
                rev.setdefault(t, []).append(s)
        dist = {s: 0 for s in self.accepting}
        q = deque(self.accepting)
        while q:
            t = q.popleft()
            for s in rev.get(t, []):
                if s not in dist:
                    dist[s] = dist[t] + 1; q.append(s)
        return dist

    def distance_to_accept(self, state: str) -> Optional[int]:
        return self._dist.get(state)

    def completion_byte(self, state: str) -> Optional[int]:
        d = self._dist[state]
        if d == 0:
            return None
        return min(b for b, t in self.T[state].items() if self._dist.get(t) == d - 1)


class TokenMasker:
    def __init__(self, dfa: CICOGrammarDFA, vocab: Sequence[bytes], eos_id: int):
        self.dfa, self.vocab, self.eos_id = dfa, list(vocab), eos_id
        self._trie: List[Dict[int, int]] = [{}]
        self._ends: List[List[int]] = [[]]
        for tid, tok in enumerate(self.vocab):
            if tid == eos_id or len(tok) == 0:
                continue
            n = 0
            for b in tok:
                nxt = self._trie[n].get(b)
                if nxt is None:
                    self._trie.append({}); self._ends.append([])
                    nxt = len(self._trie) - 1
                    self._trie[n][b] = nxt
                n = nxt
            self._ends[n].append(tid)
        self._cache: Dict[str, Tuple[int, ...]] = {}

    def allowed_token_ids(self, state: str) -> Tuple[int, ...]:
        hit = self._cache.get(state)
        if hit is not None:
            return hit
        out: List[int] = []
        stack = [(0, state)]
        while stack:
            node, q = stack.pop()
            for b, child in self._trie[node].items():
                nq = self.dfa.step(q, b)
                if nq is None:
                    continue
                out.extend(self._ends[child])
                stack.append((child, nq))
        if self.dfa.is_accepting(state):
            out.append(self.eos_id)
        res = tuple(sorted(out))
        self._cache[state] = res
        return res

    def allowed_token_ids_naive(self, state: str) -> Tuple[int, ...]:
        out = []
        for tid, tok in enumerate(self.vocab):
            if tid == self.eos_id or not tok:
                continue
            if self.dfa.run(tok, state) is not None:
                out.append(tid)
        if self.dfa.is_accepting(state):
            out.append(self.eos_id)
        return tuple(sorted(out))

    def advance(self, state: str, token_id: int) -> str:
        if token_id == self.eos_id:
            if not self.dfa.is_accepting(state):
                raise ValueError("EOS not allowed in a non-accepting state")
            return state
        nq = self.dfa.run(self.vocab[token_id], state)
        if nq is None:
            raise ValueError(f"token {token_id} {self.vocab[token_id]!r} not allowed in state {state}")
        return nq

    def logit_mask(self, state: str):
        """Additive mask as a list: 0.0 allowed, -inf blocked (convert with torch.tensor)."""
        m = [float("-inf")] * len(self.vocab)
        for t in self.allowed_token_ids(state):
            m[t] = 0.0
        return m


def build_vocab(extra_random: int = 400, seed: int = 0) -> Tuple[List[bytes], int]:
    """Synthetic BPE-like vocab: all 256 bytes, grammar fragments, prose junk, random substrings."""
    import random
    rng = random.Random(seed)
    vocab: List[bytes] = [bytes([i]) for i in range(256)]
    frags = ["ADD[", "DEL[", "ADD[(", "DEL[(", ")->(", "],", "]\n", "] ", " ADD[", "\nDEL[", "12", "100",
             "0,", ",0", "#-", "#1", ":rel", "):", "depends_on", "owns", "->", "]", "[(", "0)", "1)->(2,"]
    junk = ["the", "Sure", "Sure,", "```", "{\"", "delete", "node", " the ", "I think", "ADD", "DEL", "ADD[1", "<|", "é"]
    for s in frags + junk:
        vocab.append(s.encode())
    corpus = "ADD[1,2:3] DEL[4,5]\nADD[(1,2)->(3,4):owns#5] DEL[(0,0)->(1,1):dep#-2]"
    for _ in range(extra_random):
        i = rng.randrange(len(corpus)); j = min(len(corpus), i + rng.randint(2, 6))
        vocab.append(corpus[i:j].encode())
    seen, out = set(), []
    for t in vocab:
        if t not in seen:
            seen.add(t); out.append(t)
    out.append(b"<eos>")  # special; excluded from byte matching
    return out, len(out) - 1


def sample_delta(masker: TokenMasker, rng, max_tokens: int = 40, eos_bias: float = 0.15) -> Tuple[str, List[int]]:
    """Sample under the mask. After max_tokens, steer to acceptance via shortest completion bytes."""
    dfa, state, ids = masker.dfa, CICOGrammarDFA.START, []
    byte_tok = {tok: i for i, tok in enumerate(masker.vocab) if len(tok) == 1}
    for step in range(max_tokens * 4):
        allowed = masker.allowed_token_ids(state)
        if step >= max_tokens:
            if dfa.is_accepting(state):
                ids.append(masker.eos_id); break
            tid = byte_tok[bytes([dfa.completion_byte(state)])]
        elif masker.eos_id in allowed and rng.random() < eos_bias:
            ids.append(masker.eos_id); break
        else:
            tid = rng.choice([t for t in allowed if t != masker.eos_id] or [masker.eos_id])
            if tid == masker.eos_id:
                ids.append(tid); break
        ids.append(tid)
        state = masker.advance(state, tid)
    data = b"".join(masker.vocab[t] for t in ids if t != masker.eos_id)
    return data.decode("utf-8", errors="strict"), ids
