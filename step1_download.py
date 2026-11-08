# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Step 1a: download the ISIC 2018 Task 3 data (HAM10000) from the official ISIC
bucket listed at https://challenge.isic-archive.com/data, then unzip it.

Safe to re-run: finished downloads are skipped, half-finished ones restart.
License: CC BY-NC 4.0 (non-commercial use, with attribution).
"""
import shutil
import sys
import time
import urllib.request
import zipfile

from common import RAW, TEST_IMAGES, TRAIN_IMAGES

BASE = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
FILES = [
    "ISIC2018_Task3_Training_GroundTruth.zip",
    "ISIC2018_Task3_Training_LesionGroupings.csv",
    "ISIC2018_Task3_Test_GroundTruth.zip",
    "ISIC2018_Task3_Test_Input.zip",        # ~400 MB, 1,512 images
    "ISIC2018_Task3_Training_Input.zip",    # ~2.6 GB, 10,015 images
]


def download(name: str) -> None:
    target = RAW / name
    url = BASE + name
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    if target.exists() and target.stat().st_size == size:
        print(f"have      {name} ({size / 2**20:,.0f} MB)", flush=True)
        return
    part = target.with_suffix(target.suffix + ".part")
    print(f"download  {name} ({size / 2**20:,.0f} MB)", flush=True)
    start, done, last = time.time(), 0, 0.0
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 10:
                last = time.time()
                rate = done / max(1e-6, last - start) / 2**20
                print(f"          {done / size:6.1%}  {rate:5.1f} MB/s", flush=True)
    if part.stat().st_size != size:
        sys.exit(f"{name}: got {part.stat().st_size} bytes, expected {size}. Run again.")
    part.replace(target)


def unzip(name: str, dest, expected_jpgs: int | None) -> None:
    if expected_jpgs and dest.exists() and len(list(dest.glob("*.jpg"))) >= expected_jpgs:
        print(f"unzipped  {name}", flush=True)
        return
    print(f"unzip     {name}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW / name) as z:
        for member in z.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX"):
                continue
            filename = member.filename.rsplit("/", 1)[-1]
            if not filename or filename.startswith("."):
                continue
            with z.open(member) as src, open(dest / filename, "wb") as out:
                shutil.copyfileobj(src, out)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name)
    unzip("ISIC2018_Task3_Training_GroundTruth.zip", RAW, None)
    unzip("ISIC2018_Task3_Test_GroundTruth.zip", RAW, None)
    unzip("ISIC2018_Task3_Test_Input.zip", TEST_IMAGES, 1512)
    unzip("ISIC2018_Task3_Training_Input.zip", TRAIN_IMAGES, 10015)
    print(f"train images: {len(list(TRAIN_IMAGES.glob('*.jpg')))}, test images: {len(list(TEST_IMAGES.glob('*.jpg')))}")
    print("DOWNLOAD FINISHED", flush=True)
