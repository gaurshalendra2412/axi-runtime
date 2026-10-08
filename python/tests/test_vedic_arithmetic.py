"""Implements concept-map row 105 (tenth scan): the undated page "THE MATHEMATICS OF THE VEDIC COSMOS: A Complete Observer-Based Mathematical Framework"
(the page's metadata: published 2026-03-12, modified 2026-07-30; about 3,000 words; NO byline; authorship UNKNOWN; templated "No X. No Y." style; ten
sections). Read twice. Formulas quoted verbatim and verified on the second read: "Total Mahayuga = 432,000 x (1+2+3+4) = 432,000 x 10 = 4,320,000 years";
"One Manvantara = 71 Mahayugas = 306,720,000 years"; "One Kalpa (Day of Brahma) = 14 Manvantaras = 4,294,080,000 years"; "Width(Dvipa_n) = 2^(n-1) x W0"
(seven concentric dvipas); "Nakshatra arc = 360 deg / 27 = 13.333 deg"; "2,202 Yojanas x 14.484 km x 2 per Nimesha / 0.2 seconds = approximately 3.0 x 10^8
m/s" with the conclusion "The speed of light --- calculated from observation, correct to 3 significant figures."; "1 Yojana ~ 13-16 km" on the page itself.
The page's axioms assert a fixed flat Earth; that is contradicted by evidence and nothing here uses it.

What the tests check (the ARITHMETIC of the page, using only the page's own numbers; Rajnish to accept or reject the reading):
  1. The time-cycle numbers are internally exact. 4:3:2:1 of 10 parts of 432,000 years gives 1,728,000 / 1,296,000 / 864,000 / 432,000, sum 4,320,000;
     71 of them is 306,720,000; 14 of those is 4,294,080,000 = 994 Mahayugas. (From memory, NOT on the page, to check against a primary text before print:
     the traditional Kalpa is 1,000 Mahayugas = 4,320,000,000 years, the other 6 Mahayugas being 15 junction periods, each as long as a Satya Yuga. The
     arithmetic closes exactly: 15 x 1,728,000 = 25,920,000 = 6 x 4,320,000. So the page's Kalpa omits the junctions.)
  2. The seven dvipa widths 2^(n-1) x W0 sum to 127 x W0; the largest is 64 x W0. One free number W0.
  3. The speed-of-light line does NOT come out as the page says. With the page's own inputs it gives 318,937.68 km/s, 6.39% above the true value, which
     is correct to ONE significant figure (3), not three. The yojana that would give exactly 299,792.458 km/s with a 0.2 s nimesha is 13.61 km; the page's own range
     is 13 to 16 km, which gives 286,260 to 352,320 km/s. A nimesha of 16/75 s (from memory of popular accounts of the same verse, NOT on the page) gives
     299,004 km/s with the same yojana. Two free choices (yojana length, nimesha length) that move the answer by 23% and 6.7% are not a derivation of c.
  4. Page-internal inconsistency: the page's table has dark matter 26.8% and dark energy 68.3%, which add to 95.1%, but its summary says 96%.
  5. NUMEROLOGY GUARD. The cumulative shares of the yuga ratio 4:3:2:1 are 40, 70, 90, 100 percent. Our logged containment run (8 episodes, EXPERIMENT_RESULTS.md,
     blind apply corrupts 3, 6, 7, 8 of 8 episodes after steps 1 to 4) reads 37.5, 75, 87.5, 100 percent. All four differences are at most 5 points. That looks like a match
     and IS NOT EVIDENCE: 8 episodes can only move in steps of 12.5 points, a square-root curve and a geometric saturation curve fit equally well, and there is
     no mechanism linking a count of ages to a count of corrupted episodes. Do not write this in the article or the paper.
What they do NOT show: anything about the Vedic texts themselves (the page cites none), that the page's numbers are the traditional ones beyond the arithmetic
above, or the date of any of it."""
import math, os, sys
from fractions import Fraction
sys.path.insert(0, os.path.dirname(__file__))


def test_the_time_cycle_numbers_are_exact():
    base = 432_000
    mahayuga = base * (1 + 2 + 3 + 4)
    yugas = [base * k for k in (4, 3, 2, 1)]
    assert yugas == [1_728_000, 1_296_000, 864_000, 432_000] and sum(yugas) == mahayuga == 4_320_000
    assert [Fraction(y, mahayuga) for y in yugas] == [Fraction(k, 10) for k in (4, 3, 2, 1)]
    manvantara = 71 * mahayuga
    kalpa = 14 * manvantara
    assert manvantara == 306_720_000 and kalpa == 4_294_080_000 and kalpa == 994 * mahayuga
    junctions = 15 * yugas[0]                                              # NOT on the page; from memory, to check against a primary text
    assert junctions == 25_920_000 == 6 * mahayuga and kalpa + junctions == 1000 * mahayuga == 4_320_000_000


def test_the_seven_dvipa_widths_and_the_nakshatra_arc():
    widths = [2 ** (n - 1) for n in range(1, 8)]
    assert widths == [1, 2, 4, 8, 16, 32, 64] and sum(widths) == 127 == 2 ** 7 - 1
    assert abs(360 / 27 - 13.3333) < 1e-4 and 27 * Fraction(40, 3) == 360


def test_the_speed_of_light_line_is_a_6_percent_miss_with_two_free_parameters():
    c = 299_792.458
    page = 2202 * 14.484 * 2 / 0.2
    assert abs(page - 318_937.68) < 0.01 and abs(page / c - 1 - 0.0639) < 0.0005
    assert round(page, -4) == 320_000 and round(c, -4) == 300_000          # 3.19e5 and 3.00e5 differ already in the second significant figure
    assert abs(c * 0.2 / (2202 * 2) - 13.6146) < 0.001                      # the yojana (km) that makes the page's line exact
    lo, hi = (2202 * y * 2 / 0.2 for y in (13, 16))
    assert (round(lo), round(hi)) == (286_260, 352_320) and lo < c < hi     # the page's own range of yojanas brackets c, and a lot besides
    alt = 2202 * 14.484 * 2 / (16 / 75)                                     # nimesha 16/75 s (not on the page)
    assert abs(alt - 299_004.07) < 0.01 and abs(alt / c - 1 + 0.00263) < 0.0001
    assert abs((16 / 75) / 0.2 - 1 - 0.0667) < 0.0005                        # rounding the nimesha moves the answer by 6.7%
    assert abs(26.8 + 68.3 - 95.1) < 1e-9 and 95.1 != 96


def test_numerology_guard_the_yuga_shares_fit_the_containment_run_but_so_does_anything_smooth():
    yuga = [100 * sum((4, 3, 2, 1)[:k]) / 10 for k in (1, 2, 3, 4)]
    run = [100 * k / 8 for k in (3, 6, 7, 8)]                              # 8 episodes: the smallest possible step is 12.5 points
    assert yuga == [40, 70, 90, 100] and run == [37.5, 75.0, 87.5, 100.0]
    assert max(abs(a - b) for a, b in zip(yuga, run)) == 5.0 < 12.5
    sqrt_curve = [100 * (k / 4) ** 0.5 for k in (1, 2, 3, 4)]
    geo = [100 * (1 - 0.6 ** k) / (1 - 0.6 ** 4) for k in (1, 2, 3, 4)]
    for other in (sqrt_curve, geo):                                        # two unrelated smooth curves fit within the run's own resolution
        assert max(abs(a - b) for a, b in zip(other, run)) <= 12.5 + 1e-9
    linear = [25, 50, 75, 100]
    assert max(abs(a - b) for a, b in zip(linear, run)) > 12.5            # the match is not "anything": a straight line does not fit


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
