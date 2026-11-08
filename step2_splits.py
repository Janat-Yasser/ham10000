# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Step 2: two train/validation splits of the 10,015 training images.

  splits/lesion.csv  The honest one. Whole lesions go to one side, so no
                     validation image has a sibling photo in training.
                     Stratified by class so rare classes appear in both.
  splits/image.csv   The leaky one most tutorials use: a random split by
                     image. Kept only to measure how much it flatters the
                     validation score.

Both put ~15% of images in validation. The official test set is never
touched here.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from common import RAW, SPLITS, save_json, read_ground_truth

SEED = 42

if __name__ == "__main__":
    SPLITS.mkdir(exist_ok=True)
    df = read_ground_truth(RAW / "ISIC2018_Task3_Training_GroundTruth.csv")
    groups = pd.read_csv(RAW / "ISIC2018_Task3_Training_LesionGroupings.csv")
    df = df.merge(groups[["image", "lesion_id"]], on="image", validate="one_to_one")

    # Lesion-level: 1 fold of a 7-fold stratified group split is ~14-15% of images
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    _, val_idx = next(sgkf.split(df, df["label"], groups=df["lesion_id"]))
    lesion = df.copy()
    lesion["split"] = "train"
    lesion.loc[lesion.index[val_idx], "split"] = "val"

    # Image-level (leaky): same validation size, stratified by class, ignoring lesions
    tr, va = train_test_split(df.index, test_size=len(val_idx), stratify=df["label"], random_state=SEED)
    image = df.copy()
    image["split"] = "train"
    image.loc[va, "split"] = "val"

    report = {}
    for name, s in [("lesion", lesion), ("image", image)]:
        s.to_csv(SPLITS / f"{name}.csv", index=False)
        train_lesions = set(s.loc[s.split == "train", "lesion_id"])
        val = s[s.split == "val"]
        leaked = val["lesion_id"].isin(train_lesions)
        report[name] = {
            "train_images": int((s.split == "train").sum()),
            "val_images": int(len(val)),
            "val_images_whose_lesion_is_also_in_train": int(leaked.sum()),
            "share_leaked": round(float(leaked.mean()), 4),
            "val_class_counts": val["label"].value_counts().sort_index().tolist(),
        }
    assert report["lesion"]["val_images_whose_lesion_is_also_in_train"] == 0
    save_json(report, SPLITS / "splits_report.json")
    for k, v in report.items():
        print(f"{k:6s} train {v['train_images']:5d}  val {v['val_images']:5d}  "
              f"val images with a sibling in train: {v['val_images_whose_lesion_is_also_in_train']} ({v['share_leaked']:.1%})")
    print("SPLITS FINISHED", flush=True)
