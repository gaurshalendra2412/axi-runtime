"""The pictures on the front page (README.md), drawn from the saved run files with plain Python (no plotting library), so that every number in a picture can be checked.

Run from the repository root:   PYTHONPATH=python python3 python/tools/readme_figures.py docs/img
It writes, for light and dark pages:
  headline-distance-{light,dark}.svg   how far the held state was from the right state, step by step, first 400 requests (AXI, replies applied as written, gate only)
  cost-per-right-step-{light,dark}.svg what one step with the right state cost in each of the three ways
  headline-numbers-{light,dark}.svg   the three numbers the page opens with
  social-preview.svg                   the 1280 x 640 card for GitHub link previews (the PNG of it is made from this file with a browser; it is not generated here)
The colours are the three first slots of the validated palette (blue, orange, aqua; docs of the dataviz method), each slot always the same way of keeping the state.
This file calls no model and reads only docs/results/. It lives outside the axi package on purpose: nothing the runs use may import the result code (tests/test_million_hard_result.py checks that)."""
import os, sys
from axi.experiments import million_hard_result as R

ORDER = ("scope_m7", "blind", "gate")
NAMES = {"scope_m7": "AXI", "blind": "Replies as written", "gate": "Gate only"}
SHORT = {"scope_m7": "AXI", "blind": "As written", "gate": "Gate only"}
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", scope_m7="#2a78d6", blind="#eb6834", gate="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", scope_m7="#3987e5", blind="#d95926", gate="#199e70"),
}


def n(x) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def facts() -> dict:
    """Everything a picture shows, from the saved files."""
    run1, con = R.load(), R.load_contrast()
    cost, ac = R.analyse_cost(run1, con), R.analyse_contrast(run1, con)
    a400 = R.analyse({"rows": run1["rows"][:400], "meta": run1["meta"]})
    kd = a400["kinds"]
    right = sum(v["same_text"] + v["other_order"] for v in kd.values())
    series = {"scope_m7": [r["distance"] for r in run1["rows"][:400]], "blind": [r["distance"] for r in con["blind"]["rows"]], "gate": [r["distance"] for r in con["gate"]["rows"]]}
    assert all(len(v) == 400 for v in series.values())
    return dict(series=series, cost=cost, contrast=ac, replies_right=right, slips=400 - right, steps=400, new_offsets=ac["blind"]["new_offsets"],
                wrong_steps=400 - ac["blind"]["exact"], whole=R.analyse(run1))


def _poly(ds, x0, step, y0, unit):
    pts = []
    for i, d in enumerate(ds):
        if i in (0, len(ds) - 1) or d != ds[i - 1] or d != ds[i + 1]:
            pts.append(f"{n(x0 + i * step)},{n(y0 - d * unit)}")
    return " ".join(pts)


def distance_svg(f: dict, mode: str) -> str:
    c = THEMES[mode]; W, H = 600, 420
    x0, x1, yb, yt = 44, 470, 350, 96; step = (x1 - x0) / 399; unit = (yb - yt) / 8
    ser, e = f["series"], f["contrast"]
    title = "How far the held state was from the right state, step by step"
    desc = (f"First 400 requests of one stream. AXI: 0 items off at every step. Replies applied as written: off the right state in four stretches from steps "
            f"{', '.join(str(s) for s in f['new_offsets'])}, at most {max(ser['blind'])} items off, exactly right on {e['blind']['exact']} of 400 steps. "
            f"Gate only: never corrupted but off from step {f['new_offsets'][0]} on, at most {max(ser['gate'])} items off, exactly right on {e['gate']['exact']} of 400 steps. "
            f"The model slipped on {f['slips']} of the 400 requests; four slips, at steps {', '.join(str(s) for s in f['new_offsets'])}, each started a wrong stretch.")
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d" font-family=\'{FONT}\'>',
         f'<title id="t">{title}</title><desc id="d">{desc}</desc>', f'<rect width="{W}" height="{H}" rx="12" fill="{c["surface"]}"/>']
    kx = 36                                                                                                  # legend: a line key and the name, in ink, for every series
    for k, adv in zip(ORDER, (75, 175, 130)):
        o.append(f'<line x1="{kx}" y1="26" x2="{kx + 20}" y2="26" stroke="{c[k]}" stroke-width="2" stroke-linecap="round"/><text x="{kx + 27}" y="31" font-size="16" fill="{c["ink2"]}">{NAMES[k]}</text>')
        kx += adv + 27 + 12
    o.append(f'<text x="{x0}" y="58" font-size="14" fill="{c["muted"]}">Items off the right state</text>')
    for v in range(0, 9, 2):                                                                                 # hairline grid, ticks
        y = yb - v * unit
        o.append(f'<line x1="{x0}" y1="{n(y)}" x2="{x1}" y2="{n(y)}" stroke="{c["grid"] if v else c["axis"]}" stroke-width="1"/><text x="{x0 - 8}" y="{n(y + 5)}" font-size="14" fill="{c["muted"]}" text-anchor="end">{v}</text>')
    for t in (1, 100, 200, 300, 400):
        o.append(f'<text x="{n(x0 + (t - 1) * step)}" y="372" font-size="14" fill="{c["muted"]}" text-anchor="middle">{t}</text>')
    o.append(f'<text x="{n((x0 + x1) / 2)}" y="400" font-size="15" fill="{c["ink2"]}" text-anchor="middle">Request number</text>')
    for s in f["new_offsets"]:                                                                               # where the four wrong stretches began: the model's slip
        x = x0 + (s - 1) * step
        o.append(f'<line x1="{n(x)}" y1="{yt - 6}" x2="{n(x)}" y2="{yb}" stroke="{c["grid"]}" stroke-width="1"/><text x="{n(x)}" y="{yt - 12}" font-size="13" fill="{c["muted"]}" text-anchor="middle">slip</text>')
    for k in ("blind", "gate", "scope_m7"):                                                                  # AXI last, so that it shows on the baseline
        o.append(f'<polyline points="{_poly(ser[k], x0, step, yb, unit)}" fill="none" stroke="{c[k]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for k in ORDER:                                                                                          # end dot with a surface ring, and the end label in ink
        y = yb - ser[k][-1] * unit
        o.append(f'<circle cx="{x1}" cy="{n(y)}" r="5" fill="{c[k]}" stroke="{c["surface"]}" stroke-width="2"/><text x="{x1 + 12}" y="{n(y + 5)}" font-size="16" fill="{c["ink"]}">{SHORT[k]} · {ser[k][-1]}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def cost_svg(f: dict, mode: str) -> str:
    c = THEMES[mode]; W, H = 600, 262; cost = f["cost"]
    x0, x1 = 196, 440; top = max(cost[k]["dollars_per_exact"] for k in ORDER); ys = {"scope_m7": 78, "blind": 126, "gate": 174}
    title = "What one step with the right state cost"
    desc = (f"First 400 requests. AXI ${cost['scope_m7']['dollars_per_exact']:.4f} a step with the right state; replies applied as written ${cost['blind']['dollars_per_exact']:.4f}, "
            f"{cost['blind']['times_framework']:.1f} times as much; gate only ${cost['gate']['dollars_per_exact']:.4f}, {cost['gate']['times_framework']:.1f} times as much. Every call counted, at $0.75 in and $3.75 out per million tokens.")
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d" font-family=\'{FONT}\'>',
         f'<title id="t">{title}</title><desc id="d">{desc}</desc>', f'<rect width="{W}" height="{H}" rx="12" fill="{c["surface"]}"/>',
         f'<text x="24" y="34" font-size="17" fill="{c["ink"]}" font-weight="600">What one step with the right state cost</text>',
         f'<text x="24" y="56" font-size="14" fill="{c["muted"]}">First 400 requests, every call counted</text>']
    for k in ORDER:
        d = cost[k]["dollars_per_exact"]; w = max(4.0, (x1 - x0) * d / top); y = ys[k]
        o.append(f'<text x="24" y="{y + 6}" font-size="16" fill="{c["ink2"]}">{NAMES[k]}</text>')
        o.append(f'<path d="M{x0},{y - 12} H{n(x0 + w - 4)} Q{n(x0 + w)},{y - 12} {n(x0 + w)},{y - 8} V{y + 8} Q{n(x0 + w)},{y + 12} {n(x0 + w - 4)},{y + 12} H{x0} Z" fill="{c[k]}"/>')
        lab = f"${d:.4f}" + ("" if k == "scope_m7" else f"  ·  {cost[k]['times_framework']:.1f}×")
        o.append(f'<text x="{n(x0 + w + 10)}" y="{y + 6}" font-size="16" fill="{c["ink"]}" font-weight="600">{lab}</text>')
    o.append(f'<text x="24" y="224" font-size="13" fill="{c["muted"]}">$0.75 per million tokens in, $3.75 out (thinking counted as out).</text>')
    o.append(f'<text x="24" y="244" font-size="13" fill="{c["muted"]}">The spend was about the same; what differed was how many steps ended right.</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def numbers_svg(f: dict, mode: str) -> str:
    c = THEMES[mode]; W, H = 600, 214; w = f["whole"]; cost = f["cost"]; steps = w["steps"]
    rows = [("scope_m7", f"{steps:,} / {steps:,}", "steps with exactly the right state", f"one run, Gemini 3.8 Flash, {w['tokens']['total'] / 1e6:.1f} million tokens"),
            ("blind", f"{cost['blind']['times_framework']:.1f}×", "what a right step cost without AXI", "as written, first 400 requests"),
            ("scope_m7", f"{w['only_left_out']} / {w['only_left_out']}", "slips repaired by a fixed rule", "the model left out a service's connections")]
    title = "The three numbers"
    desc = (f"{steps:,} of {steps:,} steps with exactly the right state, in one run on Gemini 3.8 Flash with {w['tokens']['total'] / 1e6:.1f} million tokens. "
            f"A step with the right state cost {cost['blind']['times_framework']:.1f} times as much when the replies were applied as written, on the first 400 requests. "
            f"{w['only_left_out']} of {w['only_left_out']} slips, where the model left out a service's connections, repaired by a fixed rule.")
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d" font-family=\'{FONT}\'>',
         f'<title id="t">{title}</title><desc id="d">{desc}</desc>', f'<rect width="{W}" height="{H}" rx="12" fill="{c["surface"]}"/>']
    for i, (k, num, cap, sub) in enumerate(rows):
        y = 16 + i * 62
        if i: o.append(f'<line x1="24" y1="{y - 3}" x2="{W - 24}" y2="{y - 3}" stroke="{c["grid"]}" stroke-width="1"/>')
        o.append(f'<rect x="24" y="{y + 4}" width="4" height="42" rx="2" fill="{c[k]}"/>')
        o.append(f'<text x="42" y="{y + 37}" font-size="30" font-weight="700" fill="{c["ink"]}">{num}</text>')
        o.append(f'<text x="276" y="{y + 26}" font-size="17" fill="{c["ink"]}">{cap}</text><text x="276" y="{y + 46}" font-size="14" fill="{c["muted"]}">{sub}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def social_svg(f: dict) -> str:
    c = THEMES["dark"]; W, H = 1280, 640; cost = f["cost"]; e = f["contrast"]
    inner = distance_svg(f, "dark").split("\n", 2)[2].rsplit("</svg>", 1)[0]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family=\'{FONT}\'>',
         f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>',
         f'<text x="64" y="104" font-size="64" font-weight="700" fill="{c["ink"]}">AXI Runtime</text>',
         f'<text x="64" y="150" font-size="30" fill="{c["ink2"]}">Keep an AI agent\'s state exactly right, step after step.</text>',
         f'<text x="64" y="300" font-size="88" font-weight="700" fill="{c["scope_m7"]}">4,000 / 4,000</text>',
         f'<text x="64" y="346" font-size="28" fill="{c["ink"]}">steps with exactly the right state</text>',
         f'<text x="64" y="384" font-size="22" fill="{c["muted"]}">one run, Gemini 3.8 Flash, {f["whole"]["tokens"]["total"] / 1e6:.1f} million tokens</text>',
         f'<text x="64" y="520" font-size="96" font-weight="700" fill="{c["blind"]}">{cost["blind"]["times_framework"]:.1f}×</text>',
         f'<text x="64" y="568" font-size="28" fill="{c["ink"]}">what a right step cost without it</text>',
         f'<text x="64" y="604" font-size="22" fill="{c["muted"]}">replies applied as written, first 400 requests</text>',
         f'<g transform="translate(790 196) scale(0.78)">{inner}</g>', "</svg>"]
    return "\n".join(o) + "\n"


def files(f: dict) -> dict:
    out = {}
    for mode in ("light", "dark"):
        out[f"headline-distance-{mode}.svg"] = distance_svg(f, mode)
        out[f"cost-per-right-step-{mode}.svg"] = cost_svg(f, mode)
        out[f"headline-numbers-{mode}.svg"] = numbers_svg(f, mode)
    out["social-preview.svg"] = social_svg(f)
    return out


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "docs/img"
    os.makedirs(dest, exist_ok=True)
    for name, text in files(facts()).items():
        with open(os.path.join(dest, name), "w", encoding="utf-8", newline="\n") as fh: fh.write(text)
        print("wrote", os.path.join(dest, name))
