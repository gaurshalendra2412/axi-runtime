"""Implements concept-map row 75 (blog, dated 21 Jan 2026, "The Cinderella Project" - the oldest text on the blog, a collage of chat logs,
quotes and essays): "Probability of at least one error: 1 - (probability of no error)^number of steps", worked as "(0.9999)^108 =~ 0.9892"
in a game-of-telephone example (108 transmissions, 0.01% error each; the post concludes 1.08% error). Two reads agreed on those lines.
This is ENGINEERING SUPPORT for the arithmetic and for how it meets our own logged containment run; it is not a new measurement.

What these tests show:
  1. The arithmetic. 0.9999^108 = 0.98926..., which rounds to 0.9893 (the post writes 0.9892, i.e. truncated). The exact chance of at
     least one error is 1.0742%; the post's 1.08% is the linear shortcut 108 x 0.01%, which is always an upper bound (union bound).
     To keep 108 steps under a 1% failure chance each step may fail with probability at most about 9.3e-5, so 0.01% is already too loose.
  2. The same formula as a baseline for our logged containment run (EXPERIMENT_RESULTS.md, 8 episodes, blind apply corrupts 3, 6, 7, 8 of 8
     episodes by steps 1-4). With a constant per-step hazard fitted at step 1 (3/8), independent errors predict 4.9, 6.1 and 6.8 episodes at steps
     2-4; the logged counts (6, 7, 8) are within two binomial standard deviations of that, so EIGHT episodes cannot tell 'independent errors' from
     'errors that breed errors'. The data neither support nor refute independence.
What they do NOT show: that errors in agent loops are independent (they are probably not, a corrupted state makes the next request harder, but 8 episodes cannot show it); anything
about the gated policies (0% corruption at every step, by construction and in the logged run); that the post meant agents (it meant a game of telephone)."""
import math

P_STEP, STEPS = 1e-4, 108


def test_the_108_step_arithmetic():
    ok = (1 - P_STEP) ** STEPS
    assert abs(ok - 0.98926) < 1e-5 and round(ok, 4) == 0.9893 and math.floor(ok * 1e4) / 1e4 == 0.9892
    err = 1 - ok
    assert abs(err - 0.010742) < 1e-6
    assert STEPS * P_STEP == 0.0108 and STEPS * P_STEP >= err                      # the post's 1.08% is the linear upper bound
    for p in (1e-6, 1e-4, 1e-2, 0.2):                                              # union bound: 1-(1-p)^n <= n p, for any p, n
        for n in (1, 2, 10, 108):
            assert 1 - (1 - p) ** n <= n * p + 1e-15
    p_max = 1 - 0.99 ** (1 / STEPS)                                                # largest per-step failure that keeps 108 steps under 1%
    assert abs(p_max - 9.3e-5) < 1e-6 and p_max < P_STEP


def test_independence_baseline_cannot_be_told_apart_from_the_logged_run_with_8_episodes():
    n, logged = 8, {1: 3, 2: 6, 3: 7, 4: 8}                                        # 38 / 75 / 88 / 100 % of 8 episodes
    h = logged[1] / n
    for k in (2, 3, 4):
        q = 1 - (1 - h) ** k                                                       # chance an episode is corrupted by step k if hazard is constant
        mean, sd = n * q, math.sqrt(n * q * (1 - q))
        z = (logged[k] - mean) / sd
        assert abs(z) < 2, (k, mean, sd, z)
    assert [round(100 * logged[k] / n) for k in (1, 2, 3, 4)] == [38, 75, 88, 100]  # the percentages quoted in EXPERIMENT_RESULTS.md
    assert 1 / 8 * 100 == 12.5                                                      # 'percentages move in steps of 12.5 points'


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
