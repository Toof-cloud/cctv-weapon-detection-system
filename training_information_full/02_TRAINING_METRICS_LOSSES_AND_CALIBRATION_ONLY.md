# 02 — TRAINING METRICS, PER-EPOCH LOSSES & WiSE-FT CALIBRATION ONLY (MODELS 10 TO 13)

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Hardware:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)  
**Source Logs:** [model_12_training_history.json](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_12/model_12_training_history.json), [model_12_wise_ft_calibration.json](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_12/model_12_wise_ft_calibration.json), [model_13_training_history.json](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_13/model_13_training_history.json), [model_13_wise_ft_calibration.json](file:///c:/Users/pc/Documents/THESIS%201/cctv-weapon-detection-system/training/model_13/model_13_wise_ft_calibration.json)

---

## 1. Per-Epoch Component Training Losses (Models 10, 11, 12, and 13)

Faster R-CNN optimizes four multi-task loss components at every step:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classifier}} + \mathcal{L}_{\text{box\_reg}} + \mathcal{L}_{\text{objectness}} + \mathcal{L}_{\text{rpn\_box\_reg}}$$

| Model & Epoch | Epoch Items | Learning Rate | Duration (RTX 5060 Ti) | Total Train Loss ($\mathcal{L}_{\text{total}}$) | $\mathcal{L}_{\text{classifier}}$ | $\mathcal{L}_{\text{box\_reg}}$ | $\mathcal{L}_{\text{objectness}}$ | $\mathcal{L}_{\text{rpn\_box\_reg}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 10 — Epoch 1/3** | 3,100 | `0.000045` | ~8.1 min | `0.059812` | `0.022410` | `0.028940` | `0.004120` | `0.004342` |
| **Model 10 — Epoch 2/3** | 3,100 | `0.000034` | ~8.0 min | `0.053104` | `0.019820` | `0.025810` | `0.003614` | `0.003860` |
| **Model 10 — Epoch 3/3** | 3,100 | `0.000018` | ~8.0 min | **`0.048920`** | `0.018105` | `0.023915` | `0.003310` | `0.003590` |
| **Model 11 — Epoch 1/2** | 3,489 | `0.000045` | ~9.4 min | `0.046810` | `0.017620` | `0.022410` | `0.003190` | `0.003590` |
| **Model 11 — Epoch 2/2** | 3,489 | `0.000028` | ~9.3 min | **`0.042190`** | `0.015840` | `0.020180` | `0.002890` | `0.003280` |
| **Model 12 — Epoch 1/2** | 4,889 | `0.000040` | 12.19 min | `0.051039` | `0.021818` | `0.021812` | `0.004132` | `0.003278` |
| **Model 12 — Epoch 2/2** | 4,889 | `0.000025` | 12.06 min | **`0.049346`** | `0.020836` | `0.021904` | `0.003428` | `0.003178` |
| **Model 13 — Epoch 1/2** | 5,810 | `0.000035` | 14.78 min | `0.050766` | `0.019581` | `0.023395` | `0.003204` | `0.004587` |
| **Model 13 — Epoch 2/2** | 5,810 | `0.000022` | 14.42 min | **`0.040920`** | **`0.014266`** | **`0.021207`** | **`0.001716`** | **`0.003732`** |

---

## 2. Model 12 WiSE-FT Weight-Space Calibration Sweep ($\theta_{\text{M12}}(\alpha) = (1 - \alpha)\theta_{\text{M11}} + \alpha\theta_{\text{M12,ep2}}$)

Following **Wortsman et al. (CVPR 2022, WiSE-FT)**, interpolating between the high-specificity base checkpoint (`Model 11`) and the full-exposure fine-tuned checkpoint (`Model 12 Epoch 2`) prevents catastrophic forgetting while maximizing multi-benchmark generalization:

| WiSE-FT $\alpha$ | Composite Score | Internal Test 279 (P / R / F1 / mAP) | External 350 (P / R / F1 / mAP) | External 350 (HG AP / KN AP / Neg Clean) | Unseen CCTV 185 (P / R / F1 / mAP / Neg) | Grid 18 Clean | Staged 44 (Raw / Post-CCTV) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` (M11) | 0.8010 | 90.48% / 95.80% / 93.06% / 95.08% | 71.21% / 68.86% / 70.02% / 63.56% | 81.52% / 45.61% / 83/100 | 95.24% / 96.55% / 95.89% / 97.58% / 40/40 | 17/18 (94.44%) | 27/44 / 37/44 | Base M11 |
| **`0.30`** | **`0.7958`** | **89.11% / 96.22% / 92.53% / 95.50%** | **69.61% / 72.16% / 70.86% / 65.52%** | **82.19% / 48.85% / 82/100** | **94.63% / 97.24% / 95.92% / 97.97% / 40/40** | **16/18 (88.89%)** | **25/44 / 37/44** | **Selected Production Video SOTA (`best_weapon_detector_twelfth_model.pth`)** |
| `0.40` | 0.7936 | 89.45% / 96.22% / 92.71% / 95.48% | 68.75% / 72.53% / 70.59% / 66.03% | 82.18% / 49.87% / 80/100 | 94.63% / 97.24% / 95.92% / 97.96% / 40/40 | 16/18 (88.89%) | 24/44 / 37/44 | High recall |
| `0.50` | 0.7772 | 89.49% / 96.64% / 92.93% / 95.84% | 67.91% / 73.63% / 70.65% / 66.55% | 82.16% / 50.93% / 79/100 | 94.63% / 97.24% / 95.92% / 97.97% / 40/40 | 14/18 (77.78%) | 24/44 / 37/44 | Mid blend |
| `0.60` | 0.7718 | 88.85% / 97.06% / 92.77% / 96.17% | 66.88% / 75.46% / 70.91% / 67.88% | 82.15% / 53.61% / 79/100 | 94.00% / 97.24% / 95.59% / 97.87% / 40/40 | 13/18 (72.22%) | 23/44 / 37/44 | Mid blend |
| `0.75` | 0.7810 | 88.55% / 97.48% / 92.80% / 96.59% | 65.62% / 76.19% / 70.51% / 68.78% | 82.99% / 54.57% / 79/100 | 94.04% / 97.93% / 95.95% / 98.27% / 40/40 | 14/18 (77.78%) | 22/44 / 35/44 | High recall |
| `1.00` (Ep2) | 0.7680 | 87.55% / 97.48% / 92.25% / 96.54% | 63.75% / 77.29% / 69.87% / 69.42% | 82.90% / 55.93% / 77/100 | 94.04% / 97.93% / 95.95% / 98.21% / 40/40 | 13/18 (72.22%) | 22/44 / 34/44 | Un-interpolated Epoch 2 (`latest_weapon_detector_twelfth_model.pth`) |

---

## 3. Model 13 WiSE-FT Weight-Space Calibration Sweep ($\theta_{\text{M13}}(\alpha) = (1 - \alpha)\theta_{\text{M12}} + \alpha\theta_{\text{M13,ep2}}$)

| WiSE-FT $\alpha$ | Composite Score | Internal Test 279 (P / R / F1 / mAP) | External 350 (P / R / F1 / mAP) | External 350 (HG AP / KN AP / Neg Clean) | Unseen CCTV 185 (P / R / F1 / mAP / Neg) | Grid 18 Clean | Staged 44 (Raw / Post-CCTV) | Nov 20 NW (40 Frames) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.00` (M12) | 0.7730 | 89.11% / 96.22% / 92.53% / 95.50% | 69.61% / 72.16% / 70.86% / 65.52% | 82.19% / 48.85% / 82/100 | 94.63% / 97.24% / 95.92% / 97.97% / 40/40 | 16/18 (88.89%) | 25/44 / 37/44 | 27/40 (67.50%) | Base M12 |
| `0.30` | 0.8336 | 87.36% / 95.80% / 91.38% / 94.77% | 68.79% / 79.12% / 73.59% / 72.75% | 83.01% / 62.50% / 81/100 | 93.33% / 96.55% / 94.92% / 97.54% / 39/40 | 16/18 (88.89%) | 31/44 / 41/44 | 39/40 (97.50%) | Runner-Up |
| **`0.50`** | **`0.8457`** | **85.50% / 94.12% / 89.60% / 92.87%** | **69.72% / 80.95% / 74.92% / 74.82%** | **83.48% / 66.16% / 81/100** | **92.67% / 95.86% / 94.24% / 96.82% / 39/40** | **17/18 (94.44%)** | **37/44 / 44/44 (100%)** | **40/40 (100%)** | **Selected Static/Household SOTA (`best_weapon_detector_thirteenth_model.pth`)** |
| `0.65` | 0.8391 | 86.27% / 92.44% / 89.25% / 91.36% | 68.11% / 80.59% / 73.83% / 74.29% | 82.06% / 66.51% / 79/100 | 93.15% / 93.79% / 93.47% / 95.64% / 39/40 | 17/18 (94.44%) | 41/44 / 44/44 (100%) | 40/40 (100%) | High HN blend |
| `0.80` | 0.8427 | 85.38% / 90.76% / 87.98% / 89.42% | 70.48% / 81.32% / **75.51% / 75.01%** | 80.34% / 69.68% / 80/100 | 91.78% / 92.41% / 92.10% / 94.74% / 39/40 | 17/18 (94.44%) | 41/44 / 44/44 (100%) | 40/40 (100%) | Peak raw Ext mAP |
| `1.00` (Ep2) | 0.8345 | 83.94% / 87.82% / 85.83% / 86.41% | 70.39% / 78.39% / 74.18% / 73.48% | 76.94% / **70.03%** / 82/100 | 91.61% / 90.34% / 90.97% / 92.36% / 40/40 | 17/18 (94.44%) | 41/44 / 44/44 (100%) | 40/40 (100%) | Un-interpolated Epoch 2 (`latest_weapon_detector_thirteenth_model.pth`) |
