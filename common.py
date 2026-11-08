# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Shared paths, class names and small helpers for every step."""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"                       # downloaded zip/csv files
TRAIN_IMAGES = DATA / "train"            # ISIC2018_Task3_Training_Input
TEST_IMAGES = DATA / "test"              # ISIC2018_Task3_Test_Input
SPLITS = ROOT / "splits"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

# Column order of the ISIC ground-truth CSVs. Index = class id.
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis / Bowen's",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
}
MEL = CLASSES.index("MEL")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_ground_truth(csv_path: Path):
    """ISIC ground truth is one-hot (image, MEL, NV, ...). Returns a DataFrame with image + label (int)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df["label"] = df[CLASSES].values.argmax(axis=1)
    return df[["image", "label"]]


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
