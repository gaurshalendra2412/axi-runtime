"""Colab checks for Layer 2 (device masks) and Layer 3 (real torch.distributed consensus).
Run from repo root:  python python/colab/colab_layers.py      (GPU recommended; needs: pip install transformers)
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from axi.engine.cico_parser import CICOParser
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker
from axi.engine.gpu_mask import TokenDFATable, TorchMaskTable
from colab.colab_check import vocab_bytes_from_tokens


def gpt2_masker():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2"); n = len(tok); strings = [""] * n
    for t, i in tok.get_vocab().items():
        if i < n: strings[i] = t
    vocab = vocab_bytes_from_tokens(strings, tok.eos_token_id)
    dfa = CICOGrammarDFA(); return dfa, TokenMasker(dfa, vocab, tok.eos_token_id), vocab


def layer2():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print(f"[L2] device={dev}")
    dfa, m, vocab = gpt2_masker()
    t0 = time.time(); tab = TokenDFATable(dfa, m); print(f"[L2] table {tab.allowed.shape} built in {time.time()-t0:.2f}s, {tab.nbytes()/1e6:.1f} MB")
    for name, s in tab.idx.items(): assert set(np.flatnonzero(tab.allowed[s])) == set(m.allowed_token_ids(name))
    print("[L2] table == CPU masks for all", len(tab.names), "states: OK")
    gm = TorchMaskTable(tab, dev); V = tab.V
    # device-resident decode: no host sync inside the loop; parse everything at the end
    B, steps = 128, 60; g = torch.Generator(device=dev).manual_seed(0)
    toks, state, done = gm.decode(lambda t: torch.randn(B, V, device=dev, generator=g), steps, B)
    toks = toks.cpu().numpy(); done = done.cpu().numpy(); assert (state.cpu().numpy() >= 0).all()
    fin = 0
    for b in range(B):
        s = b"".join(vocab[t] for t in toks[b] if t != tab.eos).decode()
        if done[b]: CICOParser.parse_transition_delta(s); fin += 1
        else: assert dfa.run(s.encode()) is not None
    print(f"[L2] GPU decode B={B}: {fin} finished rows ALL parse with the strict parser; {B-fin} unfinished rows are live prefixes: OK")
    # timing
    def bench(fn, iters=200, warm=20):
        for _ in range(warm): fn()
        if dev == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters): fn()
        if dev == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters * 1e6
    for B in (1, 64):
        logits = torch.randn(B, V, device=dev); st = torch.randint(0, len(tab.names), (B,), device=dev); st_np = st.cpu().numpy()
        cache = {n: np.array(m.allowed_token_ids(n)) for n in tab.names}
        def cpu_roundtrip():
            x = logits.cpu().numpy()
            for b in range(B):
                keep = np.full(V, -np.inf, np.float32); ids = cache[tab.names[st_np[b]]]; keep[ids] = 0; x[b] += keep
            return torch.from_numpy(x).to(dev)
        def gpu_table(): return gm.mask_(logits.clone(), st)
        def sample_only(): return torch.multinomial(torch.softmax(logits, -1), 1)
        a, b_, c = bench(cpu_roundtrip), bench(gpu_table), bench(sample_only)
        print(f"[L2] B={B:3d} V={V}: host round-trip mask {a:9.1f} us | device table mask {b_:8.1f} us | (softmax+sample alone {c:8.1f} us) -> {a/b_:.0f}x")
    print("[L2] NOTE: these are real measurements on this machine; the RFC's '1.1-1.4 us' is below a single CUDA kernel launch.")


def _worker(rank, world, caps, steps, out):
    import torch, torch.distributed as dist
    from axi.engine.pager import Pager, OutOfPhysicalBlocks
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29533", rank=rank, world_size=world)
    p = Pager(caps[rank]); rng = random.Random(7); commits = aborts = 0
    for _ in range(steps):
        writes = rng.sample(range(25), rng.randint(0, 3)); gate_ok = rng.random() > .2
        mapped = set(p.mapped_pages()); unmaps = [v for v in rng.sample(range(25), rng.randint(0, 2)) if v in mapped and v not in writes]
        tx = p.begin_transaction(); ok = 1
        try:
            for v in writes: tx.write_page(v)
            for v in unmaps: tx.unmap_page(v)
            ok = int(tx.prepare())
        except OutOfPhysicalBlocks:
            ok = 0
        vote = torch.tensor([ok & int(gate_ok)], dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)                       # consensus: any rank says 0 -> everyone aborts
        if vote.item(): tx.commit(); commits += 1
        else: tx.rollback(); aborts += 1
    p.audit(); gathered = [None] * world; dist.all_gather_object(gathered, sorted(p.mapped_pages()))
    if rank == 0:
        json.dump({"commits": commits, "aborts": aborts, "identical": all(g == gathered[0] for g in gathered), "mapped": len(gathered[0])}, open(out, "w"))
    dist.destroy_process_group()


def layer3():
    import torch.multiprocessing as mp
    out = "/tmp/axi_dist_result.json"; world = 4
    mp.spawn(_worker, args=(world, [30, 30, 9, 14], 3000, out), nprocs=world, join=True)
    r = json.load(open(out)); print("[L3]", r)
    assert r["identical"] and r["commits"] > 100 and r["aborts"] > 100
    print(f"[L3] {world} real processes, torch.distributed all_reduce(MIN): page tables identical on every rank after 3000 steps: OK")


if __name__ == "__main__":
    for f in (layer2, layer3):
        try: f()
        except ImportError as e: print(f"[{f.__name__}] skipped:", e)
