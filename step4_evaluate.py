# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Step 4: evaluate a trained model on the official ISIC 2018 test set
(1,512 images it has never seen, from a different collection period).

    python step4_evaluate.py --split lesion
    python step4_evaluate.py --split image
    python step4_evaluate.py --compare        # after both: results/summary.md

Rules that keep the test set honest:
  - Anything that is *chosen* (the temperature for calibration, the melanoma
    decision threshold) is chosen on the validation split, then applied to
    the test set unchanged.
  - Every headline number gets a 95% bootstrap interval (2,000 resamples of
    the test images), because 1,512 images, and only ~170 melanomas, leave
    real uncertainty.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from common import CLASSES, MEL, RAW, RESULTS, RUNS, SPLITS, TEST_IMAGES, TRAIN_IMAGES, read_ground_truth, save_json
from step3_train import LesionImages, build_model, make_transforms, predict

TARGET_SPECIFICITY = 0.90   # melanoma operating point, chosen on validation
BOOTSTRAP = 2000


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    """One scalar T minimising validation NLL (Guo et al., 2017)."""
    t = torch.nn.Parameter(torch.ones(1))
    lg, yt = torch.tensor(logits), torch.tensor(y)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / t.clamp(min=0.05), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.clamp(min=0.05).item())


def ece(probs, y, bins=15):
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def metrics(probs, y, mel_threshold):
    pred = probs.argmax(1)
    recall = {c: float((pred[y == i] == i).mean()) for i, c in enumerate(CLASSES) if (y == i).any()}
    is_mel = y == MEL
    flag = probs[:, MEL] >= mel_threshold
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall_per_class": recall,
        "macro_auc": float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(len(CLASSES))))),
        "melanoma_auc": float(roc_auc_score(is_mel, probs[:, MEL])),
        "melanoma_sensitivity": float(flag[is_mel].mean()),
        "melanoma_specificity": float((~flag[~is_mel]).mean()),
    }


def bootstrap(probs, y, mel_threshold, keys, seed=0):
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < len(CLASSES):
            continue
        m = metrics(probs[i], y[i], mel_threshold)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def reliability_plot(probs_before, probs_after, y, path, title):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for probs, label, color in [(probs_before, "as trained", "#ee6c4d"), (probs_after, "after temperature scaling", "#3d5a80")]:
        conf, correct = probs.max(1), probs.argmax(1) == y
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 10:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, "o-", label=f"{label} (ECE {ece(probs, y):.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("share correct")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def confusion_plot(probs, y, path, title):
    cm = confusion_matrix(y, probs.argmax(1), labels=list(range(len(CLASSES))), normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 0.5 else "#222")
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def evaluate(split, workers, run_name=""):
    run = run_name or split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(RUNS / run / "best.pt", map_location=device, weights_only=False)
    size = ckpt["config"]["size"]
    model = build_model(False).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model"])
    _, t_eval = make_transforms(size)

    df = pd.read_csv(SPLITS / f"{split}.csv")
    val = df[df.split == "val"]
    if ckpt["config"].get("limit"):
        val = val.sample(min(max(ckpt["config"]["limit"] // 4, 8), len(val)), random_state=ckpt["config"]["seed"])
    test = read_ground_truth(RAW / "ISIC2018_Task3_Test_GroundTruth.csv")
    test = test[test.image.map(lambda n: (TEST_IMAGES / f"{n}.jpg").exists())]
    kw = dict(batch_size=64, num_workers=workers)
    val_logits, val_y = predict(model, DataLoader(LesionImages(TRAIN_IMAGES, val.image, val.label, t_eval), **kw), device)
    test_logits, test_y = predict(model, DataLoader(LesionImages(TEST_IMAGES, test.image, test.label, t_eval), **kw), device)

    # chosen on validation only
    T = fit_temperature(val_logits, val_y)
    val_p, test_p_raw, test_p = softmax(val_logits / T), softmax(test_logits), softmax(test_logits / T)
    neg = np.sort(val_p[val_y != MEL, MEL])
    mel_threshold = float(neg[int(np.ceil(TARGET_SPECIFICITY * len(neg))) - 1]) if len(neg) else 0.5

    val_m = metrics(val_p, val_y, mel_threshold)
    test_m = metrics(test_p, test_y, mel_threshold)
    ci = bootstrap(test_p, test_y, mel_threshold, ["balanced_accuracy", "macro_auc", "melanoma_auc", "melanoma_sensitivity", "melanoma_specificity"])

    RESULTS.mkdir(exist_ok=True)
    np.savez_compressed(RESULTS / f"{run}_predictions.npz", test_logits=test_logits, test_y=test_y,
                        test_images=np.array(test.image), val_logits=val_logits, val_y=val_y, temperature=T)
    out = {
        "split": split, "run": run, "seed": ckpt["config"]["seed"], "best_epoch": ckpt["epoch"], "temperature": T,
        "melanoma_threshold": mel_threshold, "target_specificity_on_val": TARGET_SPECIFICITY,
        "validation": val_m, "test": test_m, "test_95ci": ci,
        "test_ece_before": ece(test_p_raw, test_y), "test_ece_after": ece(test_p, test_y),
        "n_val": int(len(val_y)), "n_test": int(len(test_y)), "n_test_melanoma": int((test_y == MEL).sum()),
    }
    save_json(out, RESULTS / f"{run}_metrics.json")
    confusion_plot(test_p, test_y, RESULTS / f"{run}_confusion.png", f"Official test set, {split}-level split")
    reliability_plot(test_p_raw, test_p, test_y, RESULTS / f"{run}_reliability.png", f"Calibration on the test set ({split} split)")
    print(json.dumps({k: out[k] for k in ["split", "best_epoch", "temperature", "test_ece_before", "test_ece_after"]}, indent=2))
    for name, m in [("validation", val_m), ("test", test_m)]:
        print(f"{name:10s} balanced acc {m['balanced_accuracy']:.3f}  macro AUC {m['macro_auc']:.3f}  "
              f"melanoma AUC {m['melanoma_auc']:.3f}  sens {m['melanoma_sensitivity']:.3f} at spec {m['melanoma_specificity']:.3f}")
    print(f"EVALUATION FINISHED ({run})", flush=True)


def compare():
    r = {s: json.loads((RESULTS / f"{s}_metrics.json").read_text()) for s in ["lesion", "image"]}
    def f(x): return f"{x:.3f}"
    def ci(s, k): lo, hi = r[s]["test_95ci"][k]; return f"{r[s]['test'][k]:.3f} ({lo:.3f}–{hi:.3f})"
    lines = [
        "# HAM10000: does the split matter?", "",
        "Same model, same training recipe; only the train/validation split differs. "
        "The official ISIC 2018 test set (1,512 images) is untouched by both.", "",
        "| | Split by lesion (honest) | Split by image (leaky) |", "|---|---|---|",
        f"| Validation balanced accuracy | {f(r['lesion']['validation']['balanced_accuracy'])} | {f(r['image']['validation']['balanced_accuracy'])} |",
        f"| **Test** balanced accuracy (95% CI) | {ci('lesion', 'balanced_accuracy')} | {ci('image', 'balanced_accuracy')} |",
        f"| Validation − test gap | {f(r['lesion']['validation']['balanced_accuracy'] - r['lesion']['test']['balanced_accuracy'])} | {f(r['image']['validation']['balanced_accuracy'] - r['image']['test']['balanced_accuracy'])} |",
        f"| Test macro AUC | {ci('lesion', 'macro_auc')} | {ci('image', 'macro_auc')} |",
        f"| Test melanoma AUC | {ci('lesion', 'melanoma_auc')} | {ci('image', 'melanoma_auc')} |",
        f"| Test melanoma sensitivity at the val-chosen 90%-specificity threshold | {ci('lesion', 'melanoma_sensitivity')} | {ci('image', 'melanoma_sensitivity')} |",
        f"| …achieved test specificity | {ci('lesion', 'melanoma_specificity')} | {ci('image', 'melanoma_specificity')} |",
        f"| Test ECE before → after temperature scaling | {f(r['lesion']['test_ece_before'])} → {f(r['lesion']['test_ece_after'])} | {f(r['image']['test_ece_before'])} → {f(r['image']['test_ece_after'])} |",
        "",
        "Read the gap row: a validation score far above the test score means the validation set was too easy, here because of sibling photos.",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("COMPARE FINISHED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["lesion", "image"])
    p.add_argument("--compare", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--run-name", default="", help="which runs/<run-name>/ to evaluate (default: the split name)")
    a = p.parse_args()
    compare() if a.compare else evaluate(a.split, a.workers, a.run_name)
