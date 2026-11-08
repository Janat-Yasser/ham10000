# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Draws the animated banner at the top of the README (light and dark).

    python assets/make_banner.py   ->  assets/banner-light.svg, assets/banner-dark.svg

The numbers are the three-seed means in results/seeds_summary.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgkit import Figure, d  # noqa: E402

HERE = Path(__file__).resolve().parent
FACES = {"Se": "source-serif-4-latin-600-normal.woff2", "SeR": "source-serif-4-latin-400-normal.woff2",
         "In": "inter-latin-400-normal.woff2", "In6": "inter-latin-600-normal.woff2"}
THEMES = {
    "light": dict(bg="#fbf8f3", ink="#2b2118", muted="#6b5d50", faint="#a3968a", line="#e6dccf", box="#f4ede3",
                  leaky="#c2410c", honest="#0f766e", skin="#f0d2bb", skin2="#e6bea3", mole="#6b3f26", mole2="#3d2416"),
    "dark": dict(bg="#13110f", ink="#f1e9df", muted="#b5a898", faint="#7c7166", line="#2c2621", box="#1c1916",
                 leaky="#fb8a5c", honest="#3fc1ad", skin="#d9b49a", skin2="#c99d80", mole="#6b3f26", mole2="#3d2416"),
}


def mole(x, y, s, rot, t):
    """One dermatoscope 'photo' of the same mole: a round field of skin, the lesion, a hair.
    rot varies per photo, like a real repeat photo taken at another angle."""
    r = s / 2
    return (f'<g transform="translate({x} {y}) rotate({rot})">'
            f'<circle r="{r}" fill="{t["skin"]}"/>'
            f'<path d="M{-r*.5} {-r*.1} C{-r*.55} {-r*.55} {r*.1} {-r*.62} {r*.42} {-r*.3} C{r*.66} {-r*.02} {r*.4} {r*.55} {r*.02} {r*.5} '
            f'C{-r*.38} {r*.48} {-r*.46} {r*.25} {-r*.5} {-r*.1}Z" fill="{t["mole"]}"/>'
            f'<path d="M{-r*.18} {-r*.12} C{-r*.1} {-r*.3} {r*.2} {-r*.25} {r*.18} {-r*.02} C{r*.15} {r*.2} {-r*.2} {r*.16} {-r*.18} {-r*.12}Z" fill="{t["mole2"]}"/>'
            f'<path d="M{-r*.95} {r*.2} q{r*.5} {-r*.15} {r*.8} {-r*.7}" stroke="{t["mole2"]}" stroke-opacity=".55" stroke-width="1.1" fill="none"/>'
            f'<circle r="{r - 1.5}" fill="none" stroke="{t["ink"]}" stroke-opacity=".18" stroke-width="3"/>'
            f'</g>')


def banner(t, seeds):
    W, H = 880, 330
    f = Figure(W, H, t, "How much does a leaky split lie? Several photos of the same mole: split by image, some land in "
               "validation and the model has seen that mole before. Validation 0.853, test 0.716. Split by lesion, all "
               "photos of a mole stay on one side: validation 0.779, test 0.731. Means of three seeds.", FACES)
    f.add(f'<rect x="0.75" y="0.75" width="{W-1.5}" height="{H-1.5}" rx="14" fill="{t["bg"]}" stroke="{t["line"]}" stroke-width="1.5"/>')
    f.text(36, 52, "A STUDY ON HAM10000 SKIN-LESION PHOTOS", "In6", 10.5, "faint", spacing=1.6, cls="fade")
    f.text(36, 98, "How much does a", "Se", 34, "ink", cls="rise", style=d(0.1))
    f.text(36, 138, "leaky split lie?", "Se", 34, "ink", cls="rise", style=d(0.2))
    for i, line in enumerate(["Many moles were photographed more than once.",
                              "Split the photos at random and the model gets",
                              "quizzed on moles it has already studied."]):
        f.text(36, 180 + i * 21, line, "SeR", 15.5, "muted", cls="fade", style=d(0.45 + i * 0.08))
    f.text(36, 296, f"EfficientNet-B0 · official ISIC 2018 test set · mean of {seeds} seeds", "In", 11.5, "faint",
           cls="fade", style=d(0.8))

    # Right side: two rows, the same three photos sorted into TRAIN / VALIDATION boxes.
    f.css.append(".fly{animation:fly 1.1s cubic-bezier(.3,.8,.25,1) both}"
                 "@keyframes fly{from{opacity:0;transform:translate(var(--dx),0)}40%{opacity:1}to{opacity:1;transform:none}}")
    X0 = 438
    rows = [("Split by image", "leaky", [0, 0, 1], "0.853", "0.716", "the model has seen this mole", 70),
            ("Split by lesion", "honest", [0, 0, 0], "0.779", "0.731", "every photo of a mole on one side", 196)]
    for k, (name, col, side, v, te, note, y0) in enumerate(rows):
        c = t[col]
        base = 0.9 + k * 1.3
        f.text(X0, y0 - 8, name, "In6", 12.5, c, cls="fade", style=d(base))
        f.text(X0 + 116, y0 - 8, note, "In", 11, "faint", cls="fade", style=d(base + 0.9))
        for j, (bx, lab, bw) in enumerate([(X0, "TRAIN", 176), (X0 + 186, "VALIDATION", 84)]):
            f.add(f'<rect x="{bx}" y="{y0}" width="{bw}" height="74" rx="9" fill="{t["box"]}" stroke="{t["line"]}" '
                  f'stroke-width="1.2" class="fade" style="{d(base)}"/>')
            f.text(bx + 10, y0 + 68, lab, "In6", 8.5, "faint", spacing=1.2, cls="fade", style=d(base))
        slots = {0: [X0 + 33, X0 + 88, X0 + 143], 1: [X0 + 228]}
        used = {0: 0, 1: 0}
        for i, s in enumerate(side):
            x = slots[s][used[s]] if s == 0 else slots[1][0]
            used[s] += 1
            f.add(f'<g class="fly" style="--dx:-{60 + i * 30}px;{d(base + 0.2 + i * 0.18)}">{mole(x, y0 + 31, 46, [-8, 14, 33][i], t)}</g>')
        if side.count(1):   # the leaked photo gets a ring
            f.add(f'<circle cx="{X0 + 228}" cy="{y0 + 31}" r="30" fill="none" stroke="{c}" stroke-width="2" '
                  f'stroke-dasharray="4 4" class="fade" style="{d(base + 0.9)}"/>')
        # the score: validation -> test
        f.text(X0 + 402, y0 + 20, "val", "In", 10.5, "faint", "end", cls="fade", style=d(base + 1.0))
        f.text(X0 + 402, y0 + 40, v, "Se", 19, c, "end", cls="rise", style=d(base + 1.0))
        f.text(X0 + 402, y0 + 58, "test " + te, "In", 10.5, "muted", "end", cls="fade", style=d(base + 1.15))
    return f


def main():
    seeds = len(json.loads((HERE.parent / "results" / "seeds_summary.json").read_text())
                ["table"]["Validation balanced accuracy"]["lesion"]["per_seed"])
    for name, t in THEMES.items():
        (HERE / f"banner-{name}.svg").write_text(banner(t, seeds).render(), encoding="utf-8")
    print("banner written")


if __name__ == "__main__":
    main()
