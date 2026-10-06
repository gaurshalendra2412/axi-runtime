"""Layer 2 (rebuilt): device-resident grammar masks via precomputed per-DFA-state tables.

Instead of walking every token's bytes through the DFA on every step (pasted kernel), precompute ONCE:
    allowed[s, t]  : bool  token t is legal in DFA state s
    next[s, t]     : int16 DFA state after emitting t (-1 = illegal)
Per decode step on device:  logits.masked_fill_(~allowed[state], -inf);  state = next[state, token]
Zero host round-trips inside the loop. Memory = S x V x 3 bytes (37 states x 50k vocab ~ 5.6 MB).
Caveat: table size grows with the number of DFA states; this suits small grammars like CICO.
"""
import numpy as np
from .cico_decoder import CICOGrammarDFA, TokenMasker


class TokenDFATable:
    def __init__(self, dfa: CICOGrammarDFA, masker: TokenMasker, vocab_size: int = None):
        self.names = dfa.states(); self.idx = {n: i for i, n in enumerate(self.names)}
        S = len(self.names); V = vocab_size or len(masker.vocab); self.V, self.eos = V, masker.eos_id
        self.allowed = np.zeros((S, V), bool); self.next = np.full((S, V), -1, np.int16)
        for name, s in self.idx.items():
            for t in masker.allowed_token_ids(name):
                self.allowed[s, t] = True
                self.next[s, t] = s if t == masker.eos_id else self.idx[dfa.run(masker.vocab[t], name)]
        self.start = self.idx[CICOGrammarDFA.START]
        self.accepting = np.array([dfa.is_accepting(n) for n in self.names])

    def nbytes(self): return self.allowed.nbytes + self.next.nbytes


def decode_numpy(table: TokenDFATable, vocab, logits_fn, steps, batch=1):
    """Reference decode loop (host numpy) used to prove table == CPU masker. logits_fn(step) -> [B,V] float."""
    state = np.full(batch, table.start); done = np.zeros(batch, bool); out = []
    for t in range(steps):
        lg = np.where(table.allowed[state], logits_fn(t), -np.inf)
        tok = np.where(done, table.eos, lg.argmax(1))
        state = np.where(done, state, table.next[state, tok]); done |= tok == table.eos; out.append(tok)
    return np.stack(out, 1), state


class TorchMaskTable:
    """Device-side wrapper (needs torch). Everything stays on `device`."""
    def __init__(self, table: TokenDFATable, device="cuda"):
        import torch
        self.torch = torch; self.t = table
        self.allowed = torch.from_numpy(table.allowed).to(device)
        self.next = torch.from_numpy(table.next.astype(np.int64)).to(device)
        self.eos = table.eos; self.device = device

    def start_state(self, batch):
        return self.torch.full((batch,), self.t.start, dtype=self.torch.long, device=self.device)

    def mask_(self, logits, state):
        """In place. logits: [B, V'] with V' >= table.V (extra padded ids are always masked)."""
        V = self.t.V
        logits[:, :V].masked_fill_(~self.allowed[state], float("-inf"))
        if logits.shape[1] > V: logits[:, V:] = float("-inf")
        return logits

    def advance(self, state, tok):
        nxt = self.next[state, tok]
        return nxt

    def decode(self, logits_fn, steps, batch, eos_bias=8.0):
        torch = self.torch
        state = self.start_state(batch); done = torch.zeros(batch, dtype=torch.bool, device=self.device); toks = []
        for t in range(steps):
            lg = logits_fn(t).clone(); lg[:, self.eos] += eos_bias
            self.mask_(lg, state)
            tok = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(1)
            tok = torch.where(done, torch.full_like(tok, self.eos), tok)
            state = torch.where(done, state, self.advance(state, tok)); done = done | (tok == self.eos); toks.append(tok)
        return torch.stack(toks, 1), state, done
