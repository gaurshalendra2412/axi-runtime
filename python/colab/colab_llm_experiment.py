"""No-training experiment on a real open LLM (Colab T4). Run from repo root:
    pip -q install transformers
    python python/colab/colab_llm_experiment.py --n 60
Part 1: agent loop (6 modes).  Part 2: rollback vs re-prefill timing on a real KV cache.
Works with byte-level-BPE tokenizers (GPT-2 / Qwen / Llama-3 style)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from axi.engine.cico_decoder import CICOGrammarDFA, TokenMasker
from axi.engine.gpu_mask import TokenDFATable, TorchMaskTable
from axi.experiments.agent_loop import make_tasks, run_experiment, print_summary
from colab.colab_check import vocab_bytes_from_tokens


def build_hf_llm(name, device):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
    tok = AutoTokenizer.from_pretrained(name)
    try: model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16)
    except TypeError: model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    model = model.to(device).eval()
    n = len(tok); strings = [""] * n
    for t, i in tok.get_vocab().items():
        if i < n: strings[i] = t
    eos = tok.eos_token_id; vocab = vocab_bytes_from_tokens(strings, eos)
    dfa = CICOGrammarDFA(); masker = TokenMasker(dfa, vocab, eos); table = TokenDFATable(dfa, masker)
    table.allowed[table.start, eos] = False; table.next[table.start, eos] = -1          # require at least one op
    mask = TorchMaskTable(table, device)
    for s in ["ADD[1,2:3]", "DEL[(0,1)->(0,2):dep#6] DEL[3,0]", "DEL[0,1] DEL[(0,1)->(0,2):owns#4]"]:   # fail early if the vocab mapping is wrong
        st = table.start
        for t in tok.encode(s, add_special_tokens=False):
            st = table.next[st, t]; assert st >= 0, f"tokenizer/vocab mismatch on {s!r}"
    print(f"[llm] {name}: vocab={n}, logits dim={model.config.vocab_size}, eos={eos} ({tok.convert_ids_to_tokens(eos)}), table {table.nbytes()/1e6:.1f} MB")

    class CICOProc(LogitsProcessor):
        def __init__(self, prompt_len): self.pl, self.state = prompt_len, None
        def __call__(self, input_ids, scores):
            if self.state is None: self.state = mask.start_state(input_ids.shape[0])
            if input_ids.shape[1] > self.pl: self.state = mask.advance(self.state, input_ids[:, -1])
            return mask.mask_(scores, self.state)

    class HFLLM:
        def generate(self, messages, constrained, max_new_tokens=120):
            enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(device) for k, v in enc.items()}; pl = enc["input_ids"].shape[1]
            procs = LogitsProcessorList([CICOProc(pl)]) if constrained else None
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, logits_processor=procs, eos_token_id=eos, pad_token_id=eos)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            new = out[0, pl:]; return tok.decode(new, skip_special_tokens=True), int(new.shape[0]), dt
    return HFLLM(), model


def timing(model, device, lengths=(4096, 16384, 32768), spec=64, chunk=2048):
    import torch
    print("\n== Part 2: real KV-cache timing ==  GPU:", torch.cuda.get_device_name(0))
    def sync(): torch.cuda.synchronize()
    def prefill(ids, upto, past=None, start=0):
        for i in range(start, upto, chunk):
            out = model.model(input_ids=ids[:, i:min(i + chunk, upto)], past_key_values=past, use_cache=True); past = out.past_key_values
        return past
    with torch.no_grad():
        prefill(torch.randint(1000, 20000, (1, 512), device=device), 512)             # warm-up
        print(f"{'ctx tokens':>11} | {'full prefill':>13} | {'append 64 spec':>14} | {'rollback=crop':>13} | {'edit @25%: re-prefill suffix':>29}")
        for L in lengths:
            try:
                ids = torch.randint(1000, 20000, (1, L + spec), device=device)
                sync(); t = time.perf_counter(); past = prefill(ids, L); sync(); t_full = time.perf_counter() - t
                if not hasattr(past, "crop"): print("  cache has no crop(); old transformers"); return
                sync(); t = time.perf_counter(); past = prefill(ids, L + spec, past, L); sync(); t_spec = time.perf_counter() - t
                sync(); t = time.perf_counter(); past.crop(L); sync(); t_crop = time.perf_counter() - t
                sync(); t = time.perf_counter(); past.crop(L // 4); past = prefill(ids, L, past, L // 4); sync(); t_suf = time.perf_counter() - t
                print(f"{L:>11} | {t_full*1e3:>10.0f} ms | {t_spec*1e3:>11.0f} ms | {t_crop*1e6:>10.0f} us | {t_suf*1e3:>26.0f} ms")
                del past; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"{L:>11} | out of memory, skipped"); torch.cuda.empty_cache()
    print("Reading: rolling back a speculative TAIL is a cheap crop in any engine. Editing an EARLIER part of a stock causal model forces re-prefill of everything after it; avoiding that is what block independence would need model support for.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--hint-rules", action="store_true"); ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out", default="/content/axi_llm_results.json"); a = ap.parse_args()
    import torch
    device = "cuda"; llm, model = build_hf_llm(a.model, device)
    tasks = make_tasks(a.n, a.seed); print(f"\n== Part 1: agent loop, {a.n} tasks, hint_rules={a.hint_rules} ==")
    res = run_experiment(llm, tasks, a.hint_rules, progress=lambda i, n: print(f"  task {i}/{n}", end="\r"))
    print_summary(res["summary"]); json.dump(res, open(a.out, "w"), indent=1, default=str); print("\nraw transcripts saved to", a.out)
    ex = next((r for r in res["rows"] if r["kind"] == "delete_dep"), None)
    if ex: print("\nexample (delete_dep):\n  M0 text:", repr(ex["M0_unconstrained"]["text"]), "\n  M1 text:", repr(ex["M1_constrained"]["text"]), "\n  gate obstructions:", ex["M2_gate"]["obstructions"])
    if not a.skip_timing: timing(model, device)
