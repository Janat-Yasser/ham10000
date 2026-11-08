# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Step 1b: look at the data before training anything.

Writes results/explore.json, results/explore.md and two figures:
  - class_counts.png: how unbalanced the 7 classes are (train vs official test)
  - gallery.png: six random training images per class
The key number: how many images are repeat photos of a lesion that has
other photos too. Those are what leak if you split by image.
"""
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from common import CLASS_NAMES, CLASSES, RAW, RESULTS, TRAIN_IMAGES, read_ground_truth, save_json

if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    train = read_ground_truth(RAW / "ISIC2018_Task3_Training_GroundTruth.csv")
    test = read_ground_truth(RAW / "ISIC2018_Task3_Test_GroundTruth.csv")
    groups = pd.read_csv(RAW / "ISIC2018_Task3_Training_LesionGroupings.csv")
    train = train.merge(groups[["image", "lesion_id", "diagnosis_confirm_type"]], on="image", validate="one_to_one")

    per_lesion = train.groupby("lesion_id").size()
    repeated_lesions = per_lesion[per_lesion > 1]
    images_in_repeated = int(repeated_lesions.sum())
    mixed_label_lesions = int((train.groupby("lesion_id")["label"].nunique() > 1).sum())

    sizes = [Image.open(p).size for p in random.Random(0).sample(sorted(TRAIN_IMAGES.glob("*.jpg")), 50)]

    stats = {
        "train_images": len(train),
        "train_lesions": int(train["lesion_id"].nunique()),
        "test_images": len(test),
        "lesions_with_more_than_one_image": int(len(repeated_lesions)),
        "images_belonging_to_those_lesions": images_in_repeated,
        "share_of_images_with_a_sibling": round(images_in_repeated / len(train), 4),
        "max_images_for_one_lesion": int(per_lesion.max()),
        "lesions_with_conflicting_labels": mixed_label_lesions,
        "image_sizes_seen_in_sample": sorted({f"{w}x{h}" for w, h in sizes}),
        "class_counts_train": {c: int((train["label"] == i).sum()) for i, c in enumerate(CLASSES)},
        "class_counts_test": {c: int((test["label"] == i).sum()) for i, c in enumerate(CLASSES)},
        "diagnosis_confirmed_by": train["diagnosis_confirm_type"].value_counts().to_dict(),
    }
    save_json(stats, RESULTS / "explore.json")

    # class balance
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = range(len(CLASSES))
    tr = [stats["class_counts_train"][c] / len(train) * 100 for c in CLASSES]
    te = [stats["class_counts_test"][c] / len(test) * 100 for c in CLASSES]
    ax.bar([i - 0.2 for i in x], tr, 0.4, label=f"train (n={len(train):,})", color="#3d5a80")
    ax.bar([i + 0.2 for i in x], te, 0.4, label=f"official test (n={len(test):,})", color="#ee6c4d")
    ax.set_xticks(list(x), CLASSES)
    ax.set_ylabel("% of images")
    ax.set_title("Seven diagnoses, very unbalanced")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "class_counts.png", dpi=150)

    # gallery
    rng = random.Random(1)
    fig, axes = plt.subplots(len(CLASSES), 6, figsize=(9, 11))
    for r, c in enumerate(CLASSES):
        imgs = train[train["label"] == r]["image"].tolist()
        for k, name in enumerate(rng.sample(imgs, min(6, len(imgs)))):
            axes[r, k].imshow(Image.open(TRAIN_IMAGES / f"{name}.jpg"))
        for k in range(6):
            axes[r, k].axis("off")
        axes[r, 0].set_title(CLASS_NAMES[c], loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "gallery.png", dpi=110)

    lines = [
        "# HAM10000: first look",
        "",
        f"- **{stats['train_images']:,} training images** of **{stats['train_lesions']:,} lesions**; "
        f"{stats['test_images']:,} images in the official test set.",
        f"- **{stats['images_belonging_to_those_lesions']:,} training images "
        f"({stats['share_of_images_with_a_sibling']:.0%}) belong to a lesion photographed more than once** "
        f"(up to {stats['max_images_for_one_lesion']} photos of one lesion). Split by image, and these "
        "siblings end up on both sides.",
        f"- Lesions whose photos carry different labels: {stats['lesions_with_conflicting_labels']}.",
        "",
        "| Class | Train | Test |", "|---|---:|---:|",
        *[f"| {CLASS_NAMES[c]} ({c}) | {stats['class_counts_train'][c]:,} | {stats['class_counts_test'][c]:,} |" for c in CLASSES],
    ]
    (RESULTS / "explore.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("EXPLORE FINISHED", flush=True)
