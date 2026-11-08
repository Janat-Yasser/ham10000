# HAM10000: does the split matter? (several seeds)

Seeds: lesion [42, 1, 2], image [42, 1, 2]. Mean ± std over seeds; the official test set (1,512 images) is untouched by every run.

| | Split by lesion (honest) | Split by image (leaky) |
|---|---|---|
| Validation balanced accuracy | 0.779 ± 0.010 | 0.853 ± 0.005 |
| **Test** balanced accuracy | 0.731 ± 0.013 | 0.716 ± 0.018 |
| Validation − test gap | 0.048 ± 0.004 | 0.137 ± 0.018 |
| Test macro AUC | 0.958 ± 0.004 | 0.959 ± 0.006 |
| Test melanoma AUC | 0.918 ± 0.004 | 0.912 ± 0.007 |
| Test melanoma sensitivity (val-chosen 90%-spec threshold) | 0.780 ± 0.015 | 0.780 ± 0.029 |
| …achieved test specificity | 0.891 ± 0.004 | 0.863 ± 0.018 |
| Test ECE after temperature scaling | 0.022 ± 0.008 | 0.038 ± 0.001 |

- **Seed-averaged models on the test set:** lesion 0.742, image 0.739; difference (lesion − image) 95% CI [-0.027, +0.033] → no clear difference on test
- **Gap per seed:** lesion 0.046, 0.045, 0.053; image 0.132, 0.122, 0.158. Every leaky run overstates its test score by more than every honest run.

