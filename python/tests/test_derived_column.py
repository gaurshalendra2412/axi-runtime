"""Implements concept-map row 125 (twelfth scan): blog 4 Sep, the post with NO heading whose first table cell is "Written & Textual Words" (Post 293; address ".../2026/09/04/4523/"; category BLACK;
byline rajnish choubey; posted 4 Sep 01:12 UTC; the body is the line "she is here" and one table; two reads, which agree cell for cell).
What the table is, from the two reads (verbatim, cells shortened with "..." where long): ONE header row "Medium / Category | 1984 Baseline | Today (Modern Era) | Shift / Factor" and NINE body rows of
four cells. The third and fourth cells ("Today", "Shift") are EMPTY in every body row (18 empty cells). The numbers all sit in the SECOND column, one per row, and the first column holds the
category names and descriptive sentences. Read down the second column the nine values come in three groups of three:
    "Written & Textual Words (Books, Newspapers, ...)": "~10,000 - 15,000 words" / "~60,000 - 100,000+ words" / "5x to 7x increase in daily textual consumption ..."
    "Visual & Cinematic Input (Expressed as "Word-Equivalents" ...)": "~2 - 3 hours of media" / "~7 - 11+ hours of media" / "3x to 4x expansion in active visual-narrative absorption time."
    "Spoken Dialogue Volume": "~16,000+ words" / "~11,000 - 12,000 words" / "~25% to 30% reduction in raw verbal exchange."
That is 3 categories x (1984 baseline, today, shift): the cells were pasted one column to the left of where the headers say. 1984 is the year the author was born (cf. the 13-hour clock row).
No source is given for any figure, and nothing here says they are true.

The reading being tested (Rajnish to accept or reject): the third quantity of each group is DERIVED from the other two (a factor is a ratio), so the table has a constraint that can be checked without knowing
whether the figures are right: the stated factor must lie inside the ratio of the two stated ranges. Counted with nothing assumed except that a "+" is read as "about":
  1. Each stated factor lies inside the range of ratios its own two ranges allow: text 60,000-100,000 over 10,000-15,000 gives 4.0 to 10.0 (stated 5-7x); visual 7-11 over 2-3 gives 2.33 to 5.5 (stated
     3-4x); spoken 11,000-12,000 against 16,000 gives a multiplier 0.6875 to 0.75, i.e. a reduction of 25% to 31.25% (stated 25-30%).
  2. The constraint recovers the layout. Pair each baseline with ANY of the three "today" values and ANY of the three factors (3! x 3! = 36 pairings, units ignored): exactly ONE pairing satisfies the
     constraint, the consecutive one above. So the arithmetic alone says the shifted cells belong to the groups of three read down the column, and the misaligned table has only one consistent repair.
  3. The check has teeth: a stated factor outside the allowed ratios fails (for example "8x to 12x" for the text row).
What they do NOT show: that any of the figures (10,000 words a day in 1984, 11-12 hours of media today, ...) is true or sourced; that the author or the tool made the layout error (the page as stored has it);
that the read-as-"about" rule is how the author meant the "+" signs (with the "+" read as open-ended the constraint is weaker); that Today/Shift were meant to hold what the second column holds (that is the repair
the arithmetic favours, not something the page says). The general point for the runtime: a column that is a function of two others is a relation a checker can enforce (roadmap option C), the gate does not."""
import os, re, sys
from itertools import permutations
sys.path.insert(0, os.path.dirname(__file__))

GRID = [                                                                     # (cell1, cell2, cell3, cell4) exactly as read twice; long cell1 shortened
    ("Medium / Category", "1984 Baseline", "Today (Modern Era)", "Shift / Factor"),
    ("Written & Textual Words (Books, Newspapers, ...", "~10,000 – 15,000 words", "", ""),
    ("Driven by physical newspapers, letters, books, ...", "~60,000 – 100,000+ words", "", ""),
    ("Amplified exponentially by endless feeds, ...", "5x to 7x increase in daily textual consumption through screen interfaces.", "", ""),
    ("Visual & Cinematic Input (Expressed as \"Word-Equivalents\" ...", "~2 – 3 hours of media", "", ""),
    ("Limited to scheduled television broadcasts, ...", "~7 – 11+ hours of media", "", ""),
    ("Continuous multi-screen streaming, YouTube, ...", "3x to 4x expansion in active visual-narrative absorption time.", "", ""),
    ("Spoken Dialogue Volume", "~16,000+ words", "", ""),
    ("Longer, uninterrupted face-to-face conversations, ...", "~11,000 – 12,000 words", "", ""),
    ("A steady, documented multi-decade decline in spoken output ...", "~25% to 30% reduction in raw verbal exchange.", "", ""),
]
CATEGORIES = ["Written & Textual Words", "Visual & Cinematic Input", "Spoken Dialogue Volume"]


def numbers(s): return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", s)]


def parse_range(s):
    n = numbers(s)
    return (n[0], n[-1])                                                     # "~16,000+ words" is one number: (16000, 16000)


def parse_factor(s):
    n = numbers(s)
    return ("cut", n[0] / 100, n[1] / 100) if "reduction" in s else ("x", n[0], n[1])


def ratio_interval(b, t): return (t[0] / b[1], t[1] / b[0])                  # the smallest and largest today/baseline the two ranges allow


def consistent(b, t, f):
    lo, hi = ratio_interval(b, t)
    kind, a, c = f
    if kind == "cut": a, c = 1 - c, 1 - a                                    # a reduction r is a multiplier 1 - r
    return lo <= a and c <= hi


def groups():
    column = [row[1] for row in GRID[1:]]                                    # the nine values read down the second column
    out = []
    for i in range(0, 9, 3):
        out.append((parse_range(column[i]), parse_range(column[i + 1]), parse_factor(column[i + 2])))
    return out


def test_the_table_has_18_empty_cells_and_the_numbers_sit_in_the_second_column_in_three_groups_of_three():
    assert len(GRID) == 10 and all(len(r) == 4 for r in GRID)
    assert sum(1 for r in GRID[1:] for c in r[2:] if c == "") == 18         # the Today and Shift cells: columns 3 and 4 of the nine body rows
    assert all(r[1] != "" for r in GRID)                                     # every row has its number in column 2
    starts = [r[0] for r in GRID[1:] if r[0].startswith(tuple(CATEGORIES))]
    assert len(starts) == 3 and [GRID[1][0].split(" (")[0], GRID[4][0].split(" (")[0], GRID[7][0]] == CATEGORIES      # the category names open rows 2, 5 and 8
    g = groups()
    assert g[0] == ((10000.0, 15000.0), (60000.0, 100000.0), ("x", 5.0, 7.0))
    assert g[1] == ((2.0, 3.0), (7.0, 11.0), ("x", 3.0, 4.0))
    assert g[2] == ((16000.0, 16000.0), (11000.0, 12000.0), ("cut", 0.25, 0.3))


def test_each_stated_factor_lies_inside_the_ratios_its_own_two_ranges_allow():
    g = groups()
    assert [tuple(round(x, 4) for x in ratio_interval(b, t)) for b, t, f in g] == [(4.0, 10.0), (2.3333, 5.5), (0.6875, 0.75)]
    assert all(consistent(b, t, f) for b, t, f in g)
    b, t, _ = g[0]
    assert not consistent(b, t, ("x", 8.0, 12.0)) and not consistent(b, t, ("x", 2.0, 3.0))      # a factor outside 4.0..10.0 fails
    b, t, _ = g[2]
    assert not consistent(b, t, ("cut", 0.40, 0.50)) and not consistent(b, t, ("cut", 0.05, 0.10))


def test_the_constraint_recovers_the_layout_only_one_of_36_pairings_is_consistent():
    g = groups()
    base = [x[0] for x in g]; today = [x[1] for x in g]; fact = [x[2] for x in g]
    good = [(pt, pf) for pt in permutations(range(3)) for pf in permutations(range(3))
            if all(consistent(base[i], today[pt[i]], fact[pf[i]]) for i in range(3))]
    assert len(list(permutations(range(3)))) ** 2 == 36
    assert good == [((0, 1, 2), (0, 1, 2))]                                  # the consecutive reading, and nothing else
    swapped = [(base[0], today[1], fact[0]), (base[1], today[0], fact[1]), (base[2], today[2], fact[2])]
    assert not all(consistent(*row) for row in swapped)                      # for instance text baseline with visual 'today' fails


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
