"""Implements concept-map row 137 (thirteenth scan): the checkable arithmetic on eleven blog pages, so that nothing numeric from them reaches the article or the paper unchecked.
ENGINEERING SUPPORT: it checks the pages' SUMS and ROUNDINGS, not whether their assumptions are true. Quotes below were verified on a second read (YES) unless marked.
  (A) 11 Aug, "Yes, that is the entire equation, folded perfectly one last time." (Post 317, two reads): "The average distance from Earth to the Sun is 150 billion meters"; "8 billion
      people and an average arm span of 1.5 meters per person"; "the combined physical reach of humanity equals 12 billion meters"; "leaves a massive gap of 138 billion meters";
      "yields exactly:" "17.25 meters per rope"; its code uses 150_000_000_000, 8_000_000_000 and 1.5.
  (B) The blog host's storage dashboard as three pages describe it. 11 Aug (Post 371, two reads): "1,024 MB total, with 240 MB used", "(23%)", "239 Published posts, 141 Published pages",
      "That is 380 separate pieces of 2D text", "77% of the space". 12 Aug (Post 406, "THE DASHBOARD AS NOTEBOOK", two reads for these lines): "267 Published posts", "158 Published pages",
      "1,024 MB Storage Space", "257.17 MB (25%) Used". 2 Sep (Post 385, slug 4073, two reads): "2,949 published posts and 469 published pages", "1,024 MB of allowed storage space, with 469.01 MB (46%) currently used".
  (C) 12 Aug, Post 406 again (verbatim, second read): "1 MB = 1,048,576 bytes"; "1,048,576 ÷ 600 ≈ 1,747.63 pages per MB"; "1.79 million pages ÷ 200 pages per notebook = 8,948 notebooks";
      "257.17 MB used = 449,000 pages used"; "75% remaining = 1.34 million pages left"; "267 = 2 + 6 + 7 = 15 = 1 + 5 = 6"; "158 = 1 + 5 + 8 = 14 = 1 + 4 = 5"; "267 + 158 = 425"; "7 + 7 + 7 = 21".
  (D) 11 Aug, Post 372 ("If we do the rough arithmetic of this entire thread...", one read) and Post 370 (second read): "approximately 1,900 words" (the person), "approximately 17,000 words" (the reply),
      "roughly 1 : 9", "For every one word you sent, I sent nine back."
  (E) 2 Sep, Post 384 (two reads): "if a creator made 1,337 posts across the timeline, that averages out to roughly 41.7 posts per day" over a "32-day streak" (1,337 is hypothetical in the page).
  (F) 5 Sep, slug 4824 (Post 377, two reads): 26 letters (5 vowels, 21 consonants); 49 units (13 + 36); "N^L, where N is the character set size and L is word length"; lengths 3, 4, 5:
      17,576 vs 117,649; 456,976 vs 5,764,801; 11,881,376 vs 282,475,249.
  (G) 11 Aug, Post 369 and Post 370 (second reads): five timestamps 4:36 PM, 4:44 PM, 4:55 PM, 5:16 PM, 5:23 PM under a title that says "The timestamps tell the entire tide of the day".
  (H) 11 Aug, Post 368 (second read): "It is a physical altar made of 8 billion individual switches (bits) arranged into 1,024 million bytes."
What these tests show: every sum above is right to the page's own rounding; the host dashboard ROUNDS its percentages (23.4 -> 23, 76.6 -> 77, 25.1 -> 25, 45.8 -> 46) while the one AI-composed
division here (1,337 / 32 = 41.78) is TRUNCATED to 41.7 (the same habit was found in the 5 Aug and 12 Sep pages, tests/test_blog_arithmetic.py); the rope chain has a fencepost slip of about one
part in eight billion and uses 150 billion for an astronomical unit of 149.6 billion (0.27% high); "1 : 9" is 1 : 8.95; 89.9% of the words in the thread are the reply's; 47 minutes is not "the
day"; and the pages mix a binary MB (1 MB = 1,048,576 bytes, page Post 406) with a decimal "1,024 million bytes" (page Post 368), which moves "8 billion bits" to 8.59 billion if the binary reading is used.
What they do NOT show: that any of these numbers means anything beyond arithmetic; that the dashboard figures say anything about AXI (they are a hosting quota and a post count); that 784 is
special (it is 1,024 - 240; its being 28^2 is a coincidence of a subtraction: among the numbers 1..1,024, 32 are perfect squares, so a second square from a subtraction is not rare); that the digital
roots (casting out nines) carry meaning: they are the sum modulo 9 and cannot fail for a correct addition. The astronomical unit (149,597,870,700 m) is exact by the IAU 2012 definition."""
from fractions import Fraction as F

AU_M = 149_597_870_700


def dr(n):
    """Digital root (casting out nines): 1 + (n - 1) mod 9 for n >= 1."""
    return 1 + (n - 1) % 9


def test_the_rope_to_the_sun_is_17_25_metres_per_person_and_the_page_rounds_the_distance_up():
    DIST, N, ARM = F(150_000_000_000), 8_000_000_000, F(3, 2)
    human = N * ARM
    short = DIST - human
    rope = short / N
    assert (human, short, rope) == (12_000_000_000, 138_000_000_000, F(69, 4)) and float(rope) == 17.25
    assert ARM + rope == F(75, 4) == F("18.75") and N * (ARM + rope) == DIST      # N persons and N ropes, one after the other, make exactly the distance
    fencepost = short / (N - 1)                                                   # N persons need only N - 1 ropes between them if the chain starts and ends on a person
    assert 0 < (fencepost - rope) / rope < F(2, 10 ** 9)
    over = (DIST - AU_M) / AU_M
    assert F(26, 10_000) < over < F(27, 10_000)                                   # 0.27% above the astronomical unit
    rope_au = (F(AU_M) - human) / N
    assert abs(float(rope_au) - 17.25) / 17.25 < 0.003 and 17.19 < float(rope_au) < 17.20


def test_the_host_dashboard_rounds_its_percentages_and_the_posts_and_pages_add_up():
    series = [("11 Aug", F(240), 23, 239, 141, 380), ("12 Aug", F("257.17"), 25, 267, 158, 425), ("2 Sep", F("469.01"), 46, 2949, 469, 3418)]
    for _, used, pct, posts, pages, total in series:
        exact = 100 * used / 1024
        assert round(exact) == pct and posts + pages == total
        if used == F("469.01"): assert int(exact) == 45 != pct               # truncation would give 45; the dashboard says 46
    assert round(100 * F(784) / 1024) == 77 and int(100 * F(784) / 1024) == 76          # 784 = 1,024 - 240, 76.56% shown as 77%
    assert 1024 - 240 == 784 and round(100 - 100 * F("257.17") / 1024) == 75      # Post 406's "75% remaining"
    squares = [n for n in range(1, 1025) if int(n ** 0.5) ** 2 == n]
    assert len(squares) == 32 and 784 in squares and 1024 in squares              # 32 of the first 1,024 numbers are squares, so two squares are not rare
    assert 1024 - 240 == 784 and int(784 ** 0.5) == 28 and int(1024 ** 0.5) == 32


def test_the_notebook_chain_on_the_12_aug_page_is_right_to_its_own_rounding_and_the_digital_roots_cannot_fail():
    per_mb = F(1_048_576, 600)
    assert round(float(per_mb), 2) == 1747.63
    total_pages = 1024 * per_mb
    assert round(float(total_pages)) == 1_789_570 and round(float(total_pages) / 1e6, 2) == 1.79
    assert round(float(total_pages / 200)) == 8948
    used_pages = F("257.17") * per_mb
    assert round(float(used_pages), -3) == 449_000
    assert round(float(total_pages * F(3, 4)) / 1e6, 2) == 1.34
    assert (dr(267), dr(158), dr(425), dr(dr(267) + dr(158)), dr(7 + 7 + 7)) == (6, 5, 2, 2, 3) and 267 + 158 == 425
    assert all(dr(a + b) == dr(dr(a) + dr(b)) for a in range(1, 400) for b in range(1, 400))     # casting out nines holds for every correct sum
    assert dr(240 + 784) == dr(1024) == 7                                          # and the dashboard's own pair goes through it too
    assert (dr(1), dr(9), dr(18), dr(999)) == (1, 9, 9, 9)                         # a multiple of nine has root 9, not 0


def test_one_to_nine_is_one_to_8_95_and_89_9_percent_of_the_words_are_the_replys_and_a_hypothetical_division_is_truncated():
    mine, theirs = 1_900, 17_000
    assert round(theirs / mine) == 9 and 8.94 < theirs / mine < 8.95
    assert round(100 * theirs / (mine + theirs), 1) == 89.9
    assert theirs // 1000 == 17 and F(17, theirs) == F(1, 1000)                    # "the right 17 words" would be one thousandth of the reply
    q = F(1337, 32)
    assert float(q) == 41.78125 and int(float(q) * 10) / 10 == 41.7 and round(float(q), 1) == 41.8      # the page truncates to 41.7; rounding gives 41.8
    assert round(float(F(2949, 32)), 2) == 92.16                                   # the lifetime total over the same 32 days, which the page does not divide


def test_the_n_to_the_l_counts_on_the_5_sep_page_are_right_and_the_two_alphabets_split_as_the_page_says():
    assert 5 + 21 == 26 and 13 + 36 == 49
    page = {3: (17_576, 117_649), 4: (456_976, 5_764_801), 5: (11_881_376, 282_475_249)}
    for L, (a, b) in page.items():
        assert (26 ** L, 49 ** L) == (a, b)
    assert [round(49 ** L / 26 ** L, 2) for L in (3, 4, 5)] == [6.69, 12.62, 23.77]
    assert 49 ** 3 < 26 ** 4                                                       # 49-unit strings of length 3 are fewer than 26-letter strings of length 4


def test_the_five_timestamps_span_47_minutes_not_a_day_and_the_bit_count_depends_on_which_megabyte():
    t = [16 * 60 + 36, 16 * 60 + 44, 16 * 60 + 55, 17 * 60 + 16, 17 * 60 + 23]
    gaps = [b - a for a, b in zip(t, t[1:])]
    assert gaps == [8, 11, 21, 7] and sum(gaps) == 47 < 60 and round(47 / (12 * 60) * 100, 1) == 6.5
    decimal_bits = 1_024 * 10 ** 6 * 8
    binary_bits = 1_024 * 1_048_576 * 8
    assert decimal_bits == 8_192_000_000 and binary_bits == 8_589_934_592
    assert abs(decimal_bits - 8 * 10 ** 9) / decimal_bits < 0.025 and abs(binary_bits - 8 * 10 ** 9) / binary_bits > 0.065
    assert round(1_073_741_824 / 1_024_000_000, 4) == 1.0486                       # the two readings of "1,024 MB" differ by 4.9%


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
