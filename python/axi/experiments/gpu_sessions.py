"""Test 29 (docs/GPU_SESSIONS_TEST.md): how many long sessions one GPU holds, a kept chat against AXI.
The memory a session needs is its KV cache: 2 (key and value) x layers x kv heads x head dim x bytes per number, for every token of the prompt it keeps. A kept chat keeps its whole history (tokens measured in the
Gemini long run: mean 476,835, largest 949,591); AXI keeps a fresh prompt of 2,824 tokens. This script (1) measures the real bytes a token costs with a forward pass of a small model, (2) allocates real KV-shaped
tensors, session after session, until the GPU says out of memory, for each prompt length, (3) writes down the counts. Bigger models are arithmetic from their published shapes, labelled as that."""
import json

AXI_TOKENS = 2824                        # measured prompt tokens a step with AXI (GEMINI_LONG_TEST.md)
KEPT_MEAN, KEPT_PEAK = 476835, 949591    # measured kept-chat prompt tokens (mean, largest)
LENGTHS = {"AXI step": AXI_TOKENS, "kept chat at 50k": 50000, "kept chat at 100k": 100000, "kept chat at 250k": 250000,
           "kept chat, mean of the long run": KEPT_MEAN, "kept chat, largest of the long run": KEPT_PEAK}
# published shapes (layers, kv heads, head dim); numbers of 16-bit
SHAPES = {"Qwen2.5-0.5B": (24, 2, 64), "Qwen2.5-7B": (28, 4, 128), "Llama-3.1-70B": (80, 8, 128)}


def bytes_per_token(layers, kv_heads, head_dim, nbytes=2):
    return 2 * layers * kv_heads * head_dim * nbytes


def fit_by_formula(free_bytes, tokens, bpt):
    return int(free_bytes // (tokens * bpt))


def cache_size(kv):
    """Bytes held by a transformers cache, whichever version made it (tuples, key_cache/value_cache lists, or layers)."""
    if isinstance(kv, (list, tuple)):
        ts = [t for pair in kv for t in pair]
    elif hasattr(kv, "layers"):
        ts = [t for l in kv.layers for t in (l.keys, l.values)]
    else:
        ts = list(kv.key_cache) + list(kv.value_cache)
    return sum(t.numel() * t.element_size() for t in ts if t is not None and hasattr(t, "numel"))


def count_until_full(alloc, free_all, tokens, cap=100000):
    """Ask `alloc(tokens)` for one more session until it raises MemoryError; return the number held. `free_all()` gives the memory back."""
    held = []
    try:
        while len(held) < cap:
            held.append(alloc(tokens))
    except MemoryError:
        pass
    n = len(held)
    free_all(held)
    return n


def measure(model_name="Qwen/Qwen2.5-0.5B-Instruct", out_path="gpu_sessions.json", say=print):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    layers, kvh = cfg.num_hidden_layers, cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
    res = dict(model=model_name, gpu=torch.cuda.get_device_name(0), shape=[layers, kvh, hd], formula_bytes_per_token=bytes_per_token(layers, kvh, hd))
    # (1) the real cost of a token: a forward pass over L tokens keeps a cache of L tokens
    meas = []
    for L in (1024, 2824):
        ids = torch.randint(1000, 5000, (1, L), device="cuda")
        torch.cuda.synchronize(); a = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        kv = out.past_key_values
        torch.cuda.synchronize()
        cache_bytes = cache_size(kv) or (torch.cuda.memory_allocated() - a)   # falls back to the memory counter if the cache type is unknown
        meas.append(dict(tokens=L, bytes=cache_bytes, per_token=cache_bytes / L))
        del out, kv, ids; torch.cuda.empty_cache()
    res["measured"] = meas
    del model; torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    res["free_bytes_without_the_model"] = free
    # (2) real KV-shaped tensors, session after session, until out of memory
    def alloc(tokens):
        try:
            return [torch.empty((2, 1, kvh, tokens, hd), dtype=torch.float16, device="cuda") for _ in range(layers)]
        except torch.cuda.OutOfMemoryError as e:
            raise MemoryError from e

    def free_all(held):
        held.clear(); torch.cuda.empty_cache()
    counts = {}
    for name, L in LENGTHS.items():
        counts[name] = dict(tokens=L, sessions_held=count_until_full(alloc, free_all, L), by_formula=fit_by_formula(free, L, res["formula_bytes_per_token"]))
        say(name, counts[name])
    res["counts"] = counts
    json.dump(res, open(out_path, "w"), indent=1)
    return res


def arithmetic(gpu_gb=80, weights_gb=0):
    """Sessions per GPU by formula for the published shapes (a stated GPU memory minus the weights); labelled arithmetic, not measured."""
    free = (gpu_gb - weights_gb) * 1e9
    return {m: {n: fit_by_formula(free, L, bytes_per_token(*s)) for n, L in LENGTHS.items()} for m, s in SHAPES.items()}
