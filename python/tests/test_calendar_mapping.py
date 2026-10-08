"""Implements concept-map row 90 (blog 13 Sep, the post on a family arrival, "... corresponds precisely to Bhadra 27, 2083 BS, Shukla Paksha
Pratipada on the Nepali calendar ..."; read twice, the three date lines verified word for word). It is the first MAPPING in the corpus that can
be checked by computation alone, so it is also a worked example for the mapping checker on the roadmap (GEOMETRY_EXTRACTION section 6, item 3).

The page maps a Gregorian date to a Bikram Sambat (BS) date to a tithi (a lunar day), and gives these three lines:
    "September 11, 2026 (Bhadra 26 - Aunsi / Kushe Aunsi / Buwa Ko Mukh Herne Din):"
    "September 12, 2026 (Bhadra 27 - Shukla Paksha Pratipada):"
    "September 13-14, 2026 (Bhadra 28-29 - Dwitiya & Tritiya):"   "... leads straight into Hartalika Teej and Ganesh Chaturthi on September 14."
and the rule "Pratipada is the first day of the waxing moon (Shukla Paksha)".

What these tests show (a small lunar model written here, no network, no ephemeris file):
  1. The model is calibrated: it finds four new moons that I know from eclipses (6 Jan 2000, 21 Aug 2017, 14 Oct 2023, 8 Apr 2024) to within
     45 minutes. (The eclipse times are quoted from memory; the tolerance is wide on purpose.)
  2. The page's tithi labels hold. The tithi is the Moon-Sun elongation in 12-degree steps (tithi k covers 12(k-1) to 12k degrees; 1-15 waxing,
     16-30 waning, 30 = Amavasya = Aunsi). At a morning reference time in Kathmandu the model gives Amavasya on 11 Sep 2026, Pratipada on 12 Sep,
     Dwitiya on 13 Sep and Tritiya on 14 Sep, and the 4th tithi begins on the morning of 14 Sep, before midday. That is why Hartalika Teej (a
     Tritiya observance) and Ganesh Chaturthi (a Chaturthi observance) can both be dated 14 Sep. The labels do not change for any morning reference
     time between 05:00 and 06:30 Nepal time, so the result is not a borderline case.
  3. The map date -> tithi is NOT a homomorphism of "the next day". A tithi is on average 23.6 hours, a day is 24, so over a year some tithis
     are skipped at the morning reference (the step is 2) and a few are repeated (the step is 0). Over 365 days the total advance is 365 plus
     about 6. A mapping checker therefore has to be told which relation it must preserve ("order", yes; "successor", no).
What they do NOT show: the BS date (the BS calendar is a published table of month lengths and was not available here; Bhadra 27 = 12 Sep is
consistent with the usual start of Bhadra around 17 Aug but is unverified); the festival rules (Teej and Chaturthi use further rules about
sunrise, midday and local custom, which were not modelled); anything about what the dates mean for a household."""
import math
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
NPT = timezone(timedelta(hours=5, minutes=45))


def jd(dt):
    dt = dt.astimezone(UTC)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + dt.second / 60) / 60) / 24
    if m <= 2: y -= 1; m += 12
    a = y // 100; b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


# Meeus, Astronomical Algorithms ch. 47, the larger longitude terms only: (D, M, M', F, amplitude in 1e-6 degrees).
TERMS = [(0, 0, 1, 0, 6288774), (2, 0, -1, 0, 1274027), (2, 0, 0, 0, 658314), (0, 0, 2, 0, 213618), (0, 1, 0, 0, -185116),
         (0, 0, 0, 2, -114332), (2, 0, -2, 0, 58793), (2, -1, -1, 0, 57066), (2, 0, 1, 0, 53322), (2, -1, 0, 0, 45758),
         (0, 1, -1, 0, -40923), (1, 0, 0, 0, -34720), (0, 1, 1, 0, -30383), (2, 0, 0, -2, 15327), (0, 0, 1, 2, -12528),
         (0, 0, 1, -2, 10980), (4, 0, -1, 0, 10675), (0, 0, 3, 0, 10034), (4, 0, -2, 0, 8548), (2, 1, -1, 0, -7888),
         (2, 1, 0, 0, -6766), (1, 0, -1, 0, -5163), (1, 1, 0, 0, 4987), (2, -1, 1, 0, 4036), (2, 0, 2, 0, 3994), (4, 0, 0, 0, 3861)]


def elongation(dt):
    """Moon longitude minus Sun longitude in degrees [0, 360). Accurate to a few tenths of a degree, which is about an hour."""
    t = (jd(dt) + 69.0 / 86400 - 2451545.0) / 36525
    rad = math.radians
    lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t * t
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t * t
    m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t * t
    mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t * t
    f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t * t
    e = 1 - 0.002516 * t - 0.0000074 * t * t
    a1 = 119.75 + 131.849 * t; a2 = 53.09 + 479264.290 * t
    s = 0.0
    for td, tm, tmp, tf, amp in TERMS:
        s += amp * (e ** abs(tm)) * math.sin(rad(td * d + tm * m + tmp * mp + tf * f))
    s += 3958 * math.sin(rad(a1)) + 1962 * math.sin(rad(lp - f)) + 318 * math.sin(rad(a2))
    moon = lp + s / 1e6
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    ms = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    c = (1.914602 - 0.004817 * t) * math.sin(rad(ms)) + 0.019993 * math.sin(rad(2 * ms)) + 0.000289 * math.sin(rad(3 * ms))
    return (moon - (l0 + c)) % 360


def signed(dt):                       # elongation in (-180, 180]: zero at a new moon, rising through it
    x = elongation(dt)
    return x - 360 if x > 180 else x


def new_moon_near(guess):
    lo, hi = guess - timedelta(hours=60), guess + timedelta(hours=60)
    assert signed(lo) < 0 < signed(hi)
    for _ in range(60):
        mid = lo + (hi - lo) / 2
        if signed(mid) < 0: lo = mid
        else: hi = mid
    return lo


def tithi(dt): return int(elongation(dt) // 12) + 1


def morning(day, hh=5, mm=40): return datetime(day.year, day.month, day.day, hh, mm, tzinfo=NPT)


def test_the_lunar_model_finds_known_new_moons():
    known = [(datetime(2000, 1, 6, 18, 14, tzinfo=UTC), 45), (datetime(2017, 8, 21, 18, 30, tzinfo=UTC), 45),
             (datetime(2023, 10, 14, 17, 55, tzinfo=UTC), 45), (datetime(2024, 4, 8, 18, 21, tzinfo=UTC), 45)]
    for when, tol in known:
        got = new_moon_near(when)
        assert abs((got - when).total_seconds()) < tol * 60, (when, got)


def test_the_pages_tithi_labels_hold_for_11_to_14_september_2026():
    nm = new_moon_near(datetime(2026, 9, 11, 3, 27, tzinfo=UTC))
    assert datetime(2026, 9, 11, 1, 0, tzinfo=UTC) < nm < datetime(2026, 9, 11, 6, 0, tzinfo=UTC), nm       # my recollection is 03:27 UTC
    want = {11: 30, 12: 1, 13: 2, 14: 3}                          # Amavasya (Aunsi), Pratipada, Dwitiya, Tritiya
    for hh, mm in ((5, 0), (5, 40), (6, 30)):                     # any morning reference time gives the same labels
        got = {d: tithi(morning(datetime(2026, 9, d), hh, mm)) for d in want}
        assert got == want, (hh, mm, got)
    start4 = None                                                 # when does the 4th tithi (Chaturthi) begin on 14 Sep?
    t = morning(datetime(2026, 9, 14), 0, 0)
    for k in range(24 * 60):
        if tithi(t + timedelta(minutes=k)) == 4: start4 = t + timedelta(minutes=k); break
    assert start4 is not None and morning(datetime(2026, 9, 14), 5, 40) < start4 < morning(datetime(2026, 9, 14), 12, 0), start4
    assert 0 <= elongation(morning(datetime(2026, 9, 12))) < 12                                       # Pratipada = waxing, first 12 degrees
    assert all(tithi(morning(datetime(2026, 9, d))) <= 15 for d in (12, 13, 14))                      # Shukla Paksha = tithis 1..15


def test_the_date_to_tithi_map_skips_and_repeats_so_it_is_not_a_successor_homomorphism():
    day0 = datetime(2026, 1, 1)
    seq = [tithi(morning(day0 + timedelta(days=i))) for i in range(366)]
    steps = [(seq[i + 1] - seq[i]) % 30 for i in range(365)]
    assert set(steps) <= {0, 1, 2}, set(steps)
    skips, repeats = steps.count(2), steps.count(0)
    assert skips >= 5 and repeats >= 0 and 4 <= sum(s - 1 for s in steps) <= 8, (skips, repeats, sum(steps))   # about 6 more tithis than days in a year
    assert all(a <= b for a, b in zip([sum(steps[:i]) for i in range(365)], [sum(steps[:i + 1]) for i in range(365)]))   # order is preserved: it never goes backwards


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("PASS", k)
