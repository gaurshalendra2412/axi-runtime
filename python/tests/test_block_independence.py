import numpy as np
from axi.engine.block_mask import make_mask, TinyModel

# 4 blocks x 6 tokens; tokens 0,1 of each block are the interface, 2..5 are internal scratch
BLK = np.repeat(np.arange(4), 6)
IF_FIRST = np.tile([1, 1, 0, 0, 0, 0], 4)   # interface before scratch (causality alone hides scratch from it)
IF_LAST = np.tile([0, 0, 0, 0, 1, 1], 4)    # realistic: scratch first, committed interface last
IF = IF_LAST
N = len(BLK)


def run(rule, layers, edit, seed=0, iface=None):
    rng = np.random.default_rng(seed); x = rng.normal(size=(N, 16)); m = TinyModel(16, layers, seed)
    mask = make_mask(BLK, IF if iface is None else iface, rule); a = m.forward(x, mask)
    x2 = x.copy(); x2[edit] += rng.normal(size=16) * 5      # large edit
    b = m.forward(x2, mask)
    return np.abs(a - b)


def downstream(diff, block):   # rows in blocks after `block`
    return diff[BLK > block]


def test_masks_sane():
    for rule in ("pasted", "strict"):
        m = make_mask(BLK, IF, rule)
        assert not np.triu(m, 1).any()                       # causal
        assert m.diagonal().all()                            # every token sees itself -> no empty softmax rows
    s = make_mask(BLK, IF, "strict")
    for q in range(N):
        if IF[q]: assert (s[q][~IF.astype(bool)] == 0).all()  # interface queries never see internals


def test_one_layer_pasted_rule_is_exactly_independent():
    d = run("pasted", 1, edit=3)      # internal token of block 0; 1 layer: K/V of interface tokens = raw embeddings
    assert downstream(d, 0).max() == 0.0


def test_pasted_rule_with_interface_first_is_safe_only_by_causality():
    for L in (1, 4):
        assert downstream(run("pasted", L, edit=3, iface=IF_FIRST), 0).max() == 0.0


def test_multilayer_pasted_rule_LEAKS():
    leaks = [downstream(run("pasted", L, edit=3), 0).max() for L in (2, 3, 4)]
    print("   pasted-rule leakage, layers 2/3/4:", [f"{x:.3g}" for x in leaks])
    assert all(x > 1e-3 for x in leaks)


def test_multilayer_strict_rule_is_bit_exact_independent():
    for L in (1, 2, 4, 8):
        for edit in (0, 2, 3, 8, 14):                         # internal tokens in blocks 0,1,2
            blk = BLK[edit]
            d = run("strict", L, edit)
            assert downstream(d, blk).max() == 0.0, (L, edit)
            assert d[BLK < blk].size == 0 or d[BLK < blk].max() == 0.0                  # earlier blocks untouched (causality)
            assert d[edit].max() > 0                          # non-vacuous: the edited token did change


def test_editing_interface_token_does_propagate():
    d = run("strict", 3, edit=4)                              # interface token of block 0 (IF_LAST)
    assert downstream(d, 0).max() > 1e-3                      # as it must: interface IS the contract


def test_strict_rule_across_seeds():
    for s in range(20):
        assert downstream(run("strict", 3, edit=2, seed=s), 0).max() == 0.0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
