# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Step 5: combine the runs of both splits over several seeds.

One seed can't separate "this split is better" from "this run got lucky".
So each split is trained with 3 seeds (42, 1, 2), and this step reports:
  - mean ± std over seeds of every headline number, for each split,
  - the validation-minus-test gap per seed (the leak's signature),
  - a paired bootstrap on the official test images comparing the two
    splits' seed-averaged predictions (each run's probabilities after its own
    temperature scaling, averaged over seeds).

    python step5_seeds.py      ->  results/seeds_summary.md, results/seeds_gap.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASSES, RESULTS, save_json
from step4_evaluate import softmax

SPLITS_ = ["lesion", "image"]
LABEL = {"lesion": "Split by lesion (honest)", "image": "Split by image (leaky)"}
COLOR = {"lesion": "#3d5a80", "image": "#ee6c4d"}


def runs(split):
    out = []
    for f in sorted(RESULTS.glob(f"{split}*_metrics.json")):
        m = json.loads(f.read_text())
        if m["split"] != split:
            continue
        p = np.load(f.with_name(f.name.replace("_metrics.json", "_predictions.npz")), allow_pickle=True)   # our own files; image names are stored as objects
        out.append((m, softmax(p["test_logits"] / float(p["temperature"])), p["test_y"], list(p["test_images"])))
    return out


def balanced_acc(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(len(CLASSES)) if (y == c).any()]))


def main():
    R = {s: runs(s) for s in SPLITS_}
    for s in SPLITS_:
        assert R[s], f"no runs found for {s}"
    images = R["lesion"][0][3]
    for s in SPLITS_:
        for _, _, _, im in R[s]:
            assert im == images, "test image order differs between runs"
    y = R["lesion"][0][2]

    def stat(s, f):
        v = np.array([f(m) for m, *_ in R[s]])
        return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, v

    rows = [
        ("Validation balanced accuracy", lambda m: m["validation"]["balanced_accuracy"]),
        ("**Test** balanced accuracy", lambda m: m["test"]["balanced_accuracy"]),
        ("Validation − test gap", lambda m: m["validation"]["balanced_accuracy"] - m["test"]["balanced_accuracy"]),
        ("Test macro AUC", lambda m: m["test"]["macro_auc"]),
        ("Test melanoma AUC", lambda m: m["test"]["melanoma_auc"]),
        ("Test melanoma sensitivity (val-chosen 90%-spec threshold)", lambda m: m["test"]["melanoma_sensitivity"]),
        ("…achieved test specificity", lambda m: m["test"]["melanoma_specificity"]),
        ("Test ECE after temperature scaling", lambda m: m["test_ece_after"]),
    ]
    seeds = {s: [m.get("seed", 42) for m, *_ in R[s]] for s in SPLITS_}   # the first runs (seed 42) predate the "seed" field
    lines = ["# HAM10000: does the split matter? (several seeds)", "",
             f"Seeds: lesion {seeds['lesion']}, image {seeds['image']}. Mean ± std over seeds; "
             "the official test set (1,512 images) is untouched by every run.", "",
             f"| | {LABEL['lesion']} | {LABEL['image']} |", "|---|---|---|"]
    table = {}
    for name, f in rows:
        cells = []
        for s in SPLITS_:
            mu, sd, v = stat(s, f)
            table.setdefault(name, {})[s] = {"mean": mu, "std": sd, "per_seed": v.tolist()}
            cells.append(f"{mu:.3f} ± {sd:.3f}")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")

    # Seed-averaged ("ensemble") predictions, compared on the same test images
    ens = {s: np.mean([p for _, p, _, _ in R[s]], axis=0).argmax(1) for s in SPLITS_}
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        d.append(balanced_acc(y[i], ens["lesion"][i]) - balanced_acc(y[i], ens["image"][i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    ens_bacc = {s: balanced_acc(y, ens[s]) for s in SPLITS_}
    gap = {s: table["Validation − test gap"][s]["per_seed"] for s in SPLITS_}
    lines += ["",
              f"- **Seed-averaged models on the test set:** lesion {ens_bacc['lesion']:.3f}, image {ens_bacc['image']:.3f}; "
              f"difference (lesion − image) 95% CI [{lo:+.3f}, {hi:+.3f}]"
              + (" → the honest split is better" if lo > 0 else " → the leaky split is better" if hi < 0 else " → no clear difference on test"),
              f"- **Gap per seed:** lesion {', '.join(f'{g:.3f}' for g in gap['lesion'])}; "
              f"image {', '.join(f'{g:.3f}' for g in gap['image'])}. "
              + ("Every leaky run overstates its test score by more than every honest run."
                 if min(gap["image"]) > max(gap["lesion"]) else "The gaps overlap between splits."),
              ""]
    (RESULTS / "seeds_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json({"table": table, "seed_averaged_test_bacc": ens_bacc, "diff_ci": [float(lo), float(hi)]}, RESULTS / "seeds_summary.json")

    # Figure: validation -> test for every run (the slope is the leak)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    for s in SPLITS_:
        for k, (m, *_rest) in enumerate(R[s]):
            v, t = m["validation"]["balanced_accuracy"], m["test"]["balanced_accuracy"]
            ax.plot([0, 1], [v, t], "o-", color=COLOR[s], alpha=0.85, label=LABEL[s] if k == 0 else None)
    ax.set_xticks([0, 1], ["validation", "official test"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Same model, two ways to split: score on validation vs test", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "seeds_gap.png", dpi=150)
    print("\n".join(lines))
    print("SEEDS FINISHED", flush=True)


if __name__ == "__main__":
    main()
