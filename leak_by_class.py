# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Where the leak sits: per diagnosis, how many validation images of the image split have a
sibling photo in training, and how well the leaky models recognise those images compared
with the validation images that have no sibling there. Uses only saved predictions.

    python leak_by_class.py   -> results/leak_by_class.json, figures/leak_by_class.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
RUNS = ["image", "image_s1", "image_s2"]


def main():
    df = pd.read_csv("splits/image.csv")
    val, train = df[df.split == "val"], df[df.split == "train"]
    leaked = val.lesion_id.isin(set(train.lesion_id)).values
    y = val.label.values
    preds = []
    for r in RUNS:
        z = np.load(f"results/{r}_predictions.npz", allow_pickle=True)
        assert (z["val_y"] == y).all(), "validation order must match splits/image.csv"
        preds.append(z["val_logits"].argmax(1))

    def recall(mask, c):
        return float(np.mean([(p[mask] == c).mean() for p in preds]))

    def bacc(mask):
        return float(np.mean([np.mean([(p[mask & (y == c)] == c).mean() for c in range(7)]) for p in preds]))

    out = {"val_images": int(len(y)), "leaked_images": int(leaked.sum()), "per_class": {}}
    for c, name in enumerate(CLASSES):
        m = y == c
        out["per_class"][name] = {
            "val_images": int(m.sum()), "leaked": int((m & leaked).sum()),
            "share_leaked": float((m & leaked).sum() / m.sum()),
            "recall_leaked": recall(m & leaked, c), "recall_not_leaked": recall(m & ~leaked, c)}
    out["balanced_accuracy_leaked"] = bacc(leaked)
    out["balanced_accuracy_not_leaked"] = bacc(~leaked)
    out["note"] = "recalls are means over the three image-split runs (seeds 42, 1, 2)"
    json.dump(out, open("results/leak_by_class.json", "w"), indent=2)

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#2b2b2a", "#8a8980"
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
    x = np.arange(7)
    share = [out["per_class"][c]["share_leaked"] for c in CLASSES]
    ax[0].bar(x, share, width=0.6, color=blue)
    for i, s in enumerate(share):
        ax[0].text(i, s + 0.02, f"{s:.0%}", ha="center", fontsize=8, color=ink)
    ax[0].set_title("Validation images with a sibling in training", fontsize=10, color=ink, loc="left")
    ax[0].set_ylim(0, 1.25)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rl = [out["per_class"][c]["recall_leaked"] for c in CLASSES]
    rn = [out["per_class"][c]["recall_not_leaked"] for c in CLASSES]
    ax[1].bar(x - 0.17, rl, width=0.32, color=orange, label="sibling in training")
    ax[1].bar(x + 0.17, rn, width=0.32, color=blue, label="no sibling")
    ax[1].set_title("Recall of the leaky models, by image", fontsize=10, color=ink, loc="left")
    ax[1].set_ylim(0, 1.25)
    ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    for a in ax:
        a.set_xticks(x, CLASSES, fontsize=8, color=ink)
        a.tick_params(axis="y", labelsize=8, colors=muted)
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        a.grid(axis="y", color="#e4e3dd", linewidth=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(muted)
    fig.tight_layout()
    fig.savefig("figures/leak_by_class.png", dpi=200)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
