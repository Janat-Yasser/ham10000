# HAM10000: how much does a leaky split lie?

Many skin lesions in HAM10000 were photographed more than once. Split the data by **image**, as most
tutorials do, and photos of the same mole end up in both training and validation. I trained the same
model on both kinds of split, three seeds each, and scored every run on the official ISIC 2018 test set,
which no run ever touched.

![Validation vs test, three seeds per split](figures/seeds_gap.png)

**The leaky split doesn't make a worse model. It makes a validation score that lies.** Validation
overstates test balanced accuracy by **13.7 points** with the image split and by **4.8** with the lesion
split. Every leaky run overstates by more than every honest run. On the test set the two kinds of model are
indistinguishable (seed-averaged difference, 95% CI −0.027 to +0.033). So the leak doesn't hurt the model;
it hides how good the model is, and anyone picking models, epochs or thresholds on that validation set is
picking on a number that's 9 points too optimistic.

## Results

EfficientNet-B0, ImageNet-pretrained, fine-tuned 20 epochs. Mean ± std over seeds 42, 1 and 2.
Test = official ISIC 2018 Task 3 test set, 1,512 images.

| | Split by lesion (honest) | Split by image (leaky) |
|---|---|---|
| Validation balanced accuracy | 0.779 ± 0.010 | 0.853 ± 0.005 |
| **Test** balanced accuracy | 0.731 ± 0.013 | 0.716 ± 0.018 |
| Validation − test gap | **0.048 ± 0.004** | **0.137 ± 0.018** |
| Test macro AUC | 0.958 ± 0.004 | 0.959 ± 0.006 |
| Test melanoma AUC | 0.918 ± 0.004 | 0.912 ± 0.007 |
| Melanoma sensitivity at the validation-chosen 90%-specificity threshold | 0.780 ± 0.015 | 0.780 ± 0.029 |
| …specificity it actually reached on test | 0.891 ± 0.004 | 0.863 ± 0.018 |
| Test calibration error (ECE) after temperature scaling | 0.022 ± 0.008 | 0.038 ± 0.001 |

The last two rows are the leak again, in a quieter form. A threshold chosen on the leaky validation set for
90% specificity delivers 86% on test, and the temperature fitted there calibrates worse. Anything tuned on
leaked data carries the leak with it.

Per-seed numbers, with bootstrap confidence intervals, are in [`results/`](results/):
[`seeds_summary.md`](results/seeds_summary.md), and `<run>_metrics.json` for each run.

## The data, and the leak

| | |
|---|---|
| Training images | 10,015, of 7,470 lesions |
| Images whose lesion has another photo in the set | 4,501 (45%), up to 6 photos of one lesion |
| Validation images with a sibling photo in training, **image split** | 601 of 1,476 (40.7%) |
| Same, **lesion split** | 0 |
| Official test images | 1,512 |

Two thirds of the images are benign nevi, so plain accuracy would flatter any model. Everything here is
**balanced accuracy** (the mean of the per-class recalls), plus the melanoma numbers a dermatologist would
ask about: sensitivity at a fixed specificity.

![A few training images per class](figures/gallery.png)

## The model in detail (lesion split, seed 42)

| | |
|---|---|
| Test balanced accuracy | 0.744 (95% CI 0.704–0.780) |
| Macro AUC / melanoma AUC | 0.960 / 0.922 |
| Melanoma sensitivity, specificity | 0.795, 0.890 |
| ECE before → after temperature scaling | 0.084 → 0.014 |

<p>
<img src="figures/lesion_confusion.png" width="49%" alt="Confusion matrix, lesion split">
<img src="figures/lesion_reliability.png" width="44%" alt="Reliability diagram before and after temperature scaling">
</p>

The weak spot is clear in the confusion matrix: **24% of melanomas are called benign nevi**. That's the
error that matters most, and it's the main reason this is a research project and not a tool.

Temperature scaling divides every logit by one number fitted on validation. It doesn't change a single
prediction, only the confidence, and it brings "80% sure" close to "right 80% of the time".

This model is served as an API, with monitoring, in
[lesion-classifier-service](https://github.com/Janat-Yasser/lesion-classifier-service).

## How it was done

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_download.py`, `step1_explore.py` | Download ISIC 2018 Task 3, count images per lesion and class, draw the gallery |
| 2 | `step2_splits.py` | Lesion split (`StratifiedGroupKFold` on lesion id) and the leaky image split, same sizes, `random_state=42`; reports how many validation images leak |
| 3 | `step3_train.py` | EfficientNet-B0, class-weighted cross-entropy, AdamW, one warm-up epoch then cosine decay, mixed precision; keeps the epoch with the best validation balanced accuracy |
| 4 | `step4_evaluate.py` | Temperature and melanoma threshold fitted on validation, then applied once to test; bootstrap 95% CIs, confusion matrix, reliability diagram |
| 5 | `step5_seeds.py` | Mean ± std over seeds, the per-seed gaps, and a paired bootstrap on seed-averaged test predictions |

```bash
pip install -r requirements.txt
python step1_download.py && python step1_explore.py
python step2_splits.py
python step3_train.py --split lesion --seed 42 --run-name lesion
python step4_evaluate.py --split lesion --run-name lesion
# ...the same for --split image and seeds 1 and 2, then:
python step5_seeds.py
```

Each run took about 28 minutes on a Colab T4. The notebooks in [`notebooks/`](notebooks/) are the exact
Colab runs: `1_train_seed42_colab.ipynb` (seed 42; it holds the first version of steps 3 and 4, before the
`--run-name` option was added) and `2_more_seeds_colab.ipynb` (seeds 1 and 2, and the summary).
`splits/` holds both splits, so they can be reused without re-running step 2.

## Limits

- One architecture and one recipe. Three seeds measure seed noise, not how other models behave.
- HAM10000 is mostly light skin. Nothing here says how the model does on darker skin.
- Dermatoscopic images only; a phone photo of skin is out of distribution.
- **Not a medical device.** Research and portfolio work only.

## Data

ISIC 2018 Challenge Task 3 / HAM10000, CC BY-NC 4.0. The images in `figures/gallery.png` come from it.

- Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source
  dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018).
- Codella, N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the
  International Skin Imaging Collaboration (ISIC).* arXiv:1902.03368 (2019).

© 2026 Janat Yasser. All rights reserved. See [LICENSE](LICENSE).
