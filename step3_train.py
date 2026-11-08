# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Step 3: fine-tune an ImageNet-pretrained EfficientNet-B0 on one split.

    python step3_train.py --split lesion      # the honest split
    python step3_train.py --split image       # the leaky split, for comparison

Choices, and why:
  - EfficientNet-B0 (5.3M parameters) at 224x224 with mixed precision fits
    comfortably in 4 GB of GPU memory at batch size 32.
  - Class-weighted cross-entropy: nevi are ~67% of the data, so without
    weights the model can score well by mostly saying "nevus".
    Weights are 1/sqrt(class frequency), a middle ground between none and
    full inverse frequency (which over-corrects for the 115 dermatofibromas).
  - The best epoch is picked by validation *balanced accuracy* (mean recall
    over the 7 classes), not plain accuracy, for the same reason.
  - Augmentation: flips in both directions and 90-degree-ish rotations
    (a lesion has no "up"), mild crops and colour jitter.
Outputs go to runs/<split>/ (or runs/<run-name>/): best.pt, history.csv, config.json.
"""
import argparse
import csv
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import CLASSES, RUNS, SPLITS, TRAIN_IMAGES, seed_everything

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_transforms(size: int):
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    evaluate = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, evaluate


class LesionImages(Dataset):
    def __init__(self, folder, names, labels, transform):
        self.folder, self.names, self.labels, self.transform = folder, list(names), list(labels), transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.folder / f"{self.names[i]}.jpg").convert("RGB")
        return self.transform(img), self.labels[i]


def build_model(pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True).to(memory_format=torch.channels_last)).float().cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def balanced_accuracy(logits, y):
    pred = logits.argmax(1)
    recalls = [(pred[y == c] == c).mean() for c in range(len(CLASSES)) if (y == c).any()]
    return float(np.mean(recalls))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["lesion", "image"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="for quick offline tests only")
    p.add_argument("--limit", type=int, default=0, help="use only N training images (tests only)")
    p.add_argument("--run-name", default="", help="output folder under runs/ (default: the split name)")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS / (args.run_name or args.split)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLITS / f"{args.split}.csv")
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    if args.limit:
        train_df = train_df.sample(min(args.limit, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(min(max(args.limit // 4, 8), len(val_df)), random_state=args.seed)

    t_train, t_eval = make_transforms(args.size)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(LesionImages(TRAIN_IMAGES, train_df.image, train_df.label, t_train),
                              batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(LesionImages(TRAIN_IMAGES, val_df.image, val_df.label, t_eval),
                            batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    counts = np.bincount(train_df.label, minlength=len(CLASSES)).astype(float)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights = weights / (weights * counts).sum() * counts.sum()   # keeps the average loss scale
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    model = build_model(not args.no_pretrained).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    warmup = len(train_loader)   # one epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    config = {**vars(args), "device": str(device), "train_images": len(train_df), "val_images": len(val_df),
              "class_weights": [round(w, 4) for w in weights], "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        logits, ys = predict(model, val_loader, device)
        val_loss = float(nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(ys)).item())
        bacc = balanced_accuracy(logits, ys)
        acc = float((logits.argmax(1) == ys).mean())
        row = {"epoch": epoch, "train_loss": round(total / n, 4), "val_loss": round(val_loss, 4),
               "val_acc": round(acc, 4), "val_balanced_acc": round(bacc, 4), "seconds": round(time.time() - t0, 1)}
        if device.type == "cuda":
            row["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        history.append(row)
        if bacc > best:
            best = bacc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_balanced_acc": bacc, "config": config}, out / "best.pt")
            row["saved"] = True
        print(row, flush=True)

    with open(out / "history.csv", "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in history for k in r))   # keys in order of first appearance
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(history)
    print(f"TRAINING FINISHED: best validation balanced accuracy {best:.4f}", flush=True)


if __name__ == "__main__":   # needed on Windows, where DataLoader workers re-import this file
    main()
