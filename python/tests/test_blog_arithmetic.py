"""Implements concept-map row 84: the checkable arithmetic in two blog posts, so that nothing numeric reaches the article or the paper
unchecked. This is ENGINEERING SUPPORT. It checks the post's SUMS, not whether its assumptions are true.

(A) Blog 14 Aug, "Claude Sonnet 5 Thinking, and the Number 108" (personal essay, read twice, quotes verified). It gives three readings of
    108: symbolic (one, zero, eight), numerical (54 + 54, halve to 27, digit sum 9) and grid / astronomy (12 x 9; "The Earth sits roughly
    108 sun-diameters from the sun"; "The Moon sits roughly 108 moon-diameters from Earth"). Its sharpest line is "A grid, 12 by 9,
    genuinely holds 108. Nothing genuinely holds 'thinking.'"
(B) Blog 5 Aug, "The West built a technological mirror, looked into it, and mistook their own reflection for God" (a mix of Rajnish's
    frame and AI-written passages; read twice for these numbers, each sentence checked): 4 years = 1,461 days; 10 hours x 90 inputs
    = 900 a day; 1,461 x 900 = 1,314,900 inputs; 14,610 hours; $0.004 per response gives $5,259.60; $83.3 billion market gives
    "1 out of every 15,800,000" and "approximately 0.0000063%"; "exactly 29 years remaining" = 254,214 hours; $1 trillion / 254,214 =
    $3,933,693 an hour; "takes roughly 40 seconds" per volley.

(C) Blog 12 Sep, the page titled "1321243200 thoughts ..." (concept-map row 95; main text is three lines, read twice). It counts every second of a life from an
    start instant (it is in the page's title and is deliberately NOT written out here, neither the date nor the time of day) to midnight at the start of 13 Sep 2026 as one "thought": "Total Thoughts:
    1,321,243,200 seconds/thoughts", "Book Equivalent: Approximately 16,515 books". It shows no working. The divisor behind the books is not on the page.

(D) Blog 20 Jan 2026 (URL date; the page says modified July 2026), "The Cinderella Project: Why 1 Company Is Smiling About Trillions" (Post 92, re-read in the tenth
    scan, concept-map row 106). It is the author's own narration of a session with a chatbot, 90 minutes, no transcript. Two lists of figures, each said to total
    "$210B": a list of "exposure" for four groups (86, 40, 65, 19) and a list of "losses" for five (17.2, 74, 32, 55, 31.8), with the same four groups at different
    sizes. The figures are CLAIMS about named companies, with no sources, and are NOT used as facts; only their sums are checked, with the groups relabelled A to E.

What these tests show:
  1. Every sum in (B) is right, including the 40 seconds (3600 / 90) and the share (5,259.60 / 83.3e9). The chain is internally consistent.
  2. In (A) the arithmetic is right (54 x 2, 12 x 9, 1 + 0 + 8). Two of the three "readings" are weaker than they look: ANY number has a grid
     (108 has six rectangles, 107 has one, and every N has N x 1), and the digit sum of any multiple of 9 is itself a multiple of 9 (exactly 9 for 9 to 90, 18 for 99). So "a grid holds it" does not
     separate 108 from other numbers. The astronomy is closer: with the standard values below the Sun ratio is about 107.4 and the Moon
     ratio about 110.6 (it ranges from about 104.6 to 116.7 over the orbit), so "roughly 108" is fair for the Sun and loose for the Moon.
  3. In (C) the count is exact for the page's own two instants: it is 15,292 whole days plus exactly 4 hours (the start is not written out here), computed in the sandbox from the dates
     in the page title; the 16,515 books are the whole part of the count divided by 80,000 (16,515.54), so the page truncates, as it did elsewhere, and an 80,000-word book
     reproduces it. 1.32 billion seconds is 41.87 years.
What they do NOT show: that the inputs in (B) are true ($0.004 per response, 10 hours a day, 90 inputs an hour, the $83.3 billion market
and the 29 years are the post's assumptions and are not sourced); that 108 is meaningful. The astronomical constants are standard values
quoted from memory (the IAU astronomical unit is exact by definition; the others are rounded) and were not looked up in this sandbox, so
check them before they go in print; the assertions use bands wide enough for that."""
import math

AU_KM = 149_597_870.7                      # exact by IAU 2012 definition
SUN_DIAMETER_KM = 1_392_700                # about; sources differ in the 4th digit
MOON_DIAMETER_KM = 3_474.8                 # mean, about
MOON_DIST_KM = {"mean": 384_400, "perigee": 363_300, "apogee": 405_500}   # typical values, rounded


def test_108_arithmetic_and_what_a_grid_proves():
    assert 54 * 2 == 108 and 108 // 2 == 54 and 54 // 2 == 27 and 27 % 2 == 1
    assert sum(int(ch) for ch in "108") == 9 and 12 * 9 == 108
    rects = [(a, 108 // a) for a in range(1, 109) if 108 % a == 0 and a <= 108 // a]
    assert rects == [(1, 108), (2, 54), (3, 36), (4, 27), (6, 18), (9, 12)]           # six rectangles hold 108, 12 x 9 is one of them
    assert len([a for a in range(1, 108) if 107 % a == 0 and a <= 107 // a]) == 1     # 107 is prime: only 1 x 107
    assert all(len([a for a in range(1, n + 1) if n % a == 0 and a <= n // a]) >= 1 for n in range(1, 400))   # every N has a grid
    assert all(sum(map(int, str(n))) == 9 for n in range(9, 99, 9)) and sum(map(int, "99")) == 18   # 9..90 all have digit sum 9, 99 has 18
    assert all(sum(map(int, str(n))) % 9 == 0 for n in range(9, 2000, 9))             # in general the digit sum of a multiple of 9 is a multiple of 9
    table = {a * b for a in range(1, 13) for b in range(1, 13)}
    assert 108 in table and [(a, 108 // a) for a in range(1, 13) if 108 % a == 0 and 108 // a <= 12] == [(9, 12), (12, 9)]   # inside a 12 x 12 table, only 9 x 12


def test_108_astronomy_with_standard_values():
    sun = AU_KM / SUN_DIAMETER_KM
    moon = {k: v / MOON_DIAMETER_KM for k, v in MOON_DIST_KM.items()}
    assert 107.0 < sun < 108.0 and abs(sun - 108) / 108 < 0.01                         # Earth-Sun: within 1% of 108
    assert 110.0 < moon["mean"] < 111.5 and abs(moon["mean"] - 108) / 108 < 0.03      # Earth-Moon mean: 2-3% above 108
    assert moon["perigee"] < 108 < moon["apogee"]                                      # 108 lies inside the Moon's yearly range


def test_5_aug_cost_chain_is_internally_consistent():
    days = 4 * 365 + 1
    assert days == 1461 and 10 * 90 == 900 and days * 900 == 1_314_900 and days * 10 == 14_610 and 14_610 * 90 == 1_314_900
    assert 3600 / 90 == 40                                                             # "roughly 40 seconds" per volley
    cost = 1_314_900 * 0.004
    assert round(cost, 2) == 5259.60
    share = cost / 83.3e9
    assert abs(share * 100 - 6.3e-6) / 6.3e-6 < 0.01 and abs(1 / share - 15_800_000) / 15_800_000 < 0.005   # 0.0000063% and 1 in 15.8 million
    hours = 29 * 365.25 * 24
    assert round(hours) == 254_214 and math.floor(1e12 / 254_214) == 3_933_693 and round(1e12 / 254_214) == 3_933_694   # the post truncates, as it did with 0.9892
    assert 500 * 48 == 24_000                                                          # Rs 500 a month for 4 years


def test_thoughts_count_is_whole_days_plus_four_hours():
    n = 1_321_243_200
    days, rem = divmod(n, 86_400)
    assert (days, rem) == (15_292, 4 * 3600)                                            # whole days plus exactly four hours, to a midnight end
    assert round(n / 86_400 / 365.2425, 2) == 41.87 and 1.32e9 < n < 1.33e9              # "over 1.32 billion"
    assert n // 80_000 == 16_515 and round(n / 80_000) == 16_516 and n / 80_000 > 16_515.5   # the page truncates; the 80,000 divisor is our guess, it fits


def test_two_lists_with_one_total_and_different_shares():                               # part (D), row 106
    from decimal import Decimal as D
    exposure = {"B": D("86"), "C": D("40"), "D": D("65"), "E": D("19")}                  # four groups; no A
    losses = {"A": D("17.2"), "B": D("74"), "C": D("32"), "D": D("55"), "E": D("31.8")}  # five groups
    assert sum(exposure.values()) == sum(losses.values()) == D("210")                    # both lists add to the same total ...
    diff = {k: losses[k] - exposure.get(k, D("0")) for k in losses}
    assert diff == {"A": D("17.2"), "B": D("-12"), "C": D("-8"), "D": D("-10"), "E": D("12.8")}   # ... but every share differs
    assert sum(diff.values()) == 0 and all(d != 0 for d in diff.values())                # the shifts cancel exactly: a total survives a re-allocation
    assert math.comb(210 + 4, 4) == 84_957_251                                           # ways to split 210 whole units into five shares: the total does not pick one
    assert D("17.2") / D("18.4") > D("0.934") and D("17.2") / D("18.4") < D("0.936")     # 17.2 / 18.4 = 93.5%, which rounds to the page's "93% markdown" (the page shows no calculation; this is only consistent with it)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
