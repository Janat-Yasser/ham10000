# HAM10000: does the split matter?

Same model, same training recipe; only the train/validation split differs. The official ISIC 2018 test set (1,512 images) is untouched by both.

| | Split by lesion (honest) | Split by image (leaky) |
|---|---|---|
| Validation balanced accuracy | 0.790 | 0.858 |
| **Test** balanced accuracy (95% CI) | 0.744 (0.704–0.780) | 0.727 (0.687–0.764) |
| Validation − test gap | 0.046 | 0.132 |
| Test macro AUC | 0.960 (0.949–0.969) | 0.953 (0.940–0.964) |
| Test melanoma AUC | 0.922 (0.898–0.944) | 0.906 (0.881–0.929) |
| Test melanoma sensitivity at the val-chosen 90%-specificity threshold | 0.795 (0.738–0.852) | 0.760 (0.696–0.820) |
| …achieved test specificity | 0.890 (0.874–0.907) | 0.872 (0.854–0.889) |
| Test ECE before → after temperature scaling | 0.084 → 0.014 | 0.089 → 0.038 |

Read the gap row: a validation score far above the test score means the validation set was too easy, here because of sibling photos.
