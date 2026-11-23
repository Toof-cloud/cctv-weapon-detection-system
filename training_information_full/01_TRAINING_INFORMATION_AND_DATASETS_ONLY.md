# 01 — TRAINING INFORMATION, ARCHITECTURE & DATASET CONFIGURATIONS ONLY (MODELS 1 TO 13)

**System Name:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
**Academic Unit:** National University – Manila | CCIT – Computer Science Department  
**Hardware Environment:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), PyTorch Mixed-Precision (`FP16` AMP)  

---

## 1. Detector Architecture & Hyperparameter Configuration Across Models 1–13

All 13 models use **Faster R-CNN with a ResNet-50 FPN v2 backbone** (`torchvision.models.detection.fasterrcnn_resnet50_fpn_v2`), predicting 3 classes:
- `Class 0`: Background / Hard Negative (`__background__`)
- `Class 1`: `handgun`
- `Class 2`: `knife`

| Configuration Parameter | Models 1–4 (Legacy Web/Macro) | Models 5–6 (Early CCTV Pivot) | Model 7 (Micro-Anchor Test) | Model 8 (Balanced 5-Stage) | Model 9 (128px Clamped) | **Models 10–13 (Final SOTA Architecture)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Backbone & FPN** | ResNet-50 FPN v2 | ResNet-50 FPN v2 | ResNet-50 FPN v2 | ResNet-50 FPN v2 | ResNet-50 FPN v2 | **ResNet-50 FPN v2 (5 FPN levels)** |
| **RPN Anchor Scales (px)** | `(32, 64, 128, 256, 512)` | `(32, 64, 128, 256, 512)` | `(8, 16, 32, 64, 128)` | `(16, 32, 64, 128, 256)` | `(8, 16, 32, 64, 128)` | **`(16, 32, 64, 128, 256)`** |
| **Anchor Aspect Ratios** | `(0.5, 1.0, 2.0)` | `(0.5, 1.0, 2.0)` | `(0.5, 0.75, 1.0, 1.5, 2.0)` | `(0.5, 1.0, 2.0)` | `(0.5, 0.75, 1.0, 1.5, 2.0)` | **`(0.5, 1.0, 2.0)`** |
| **RPN & RoI Box Clamping** | Default `ln(1000/16)` | Default `ln(1000/16)` | `ln(1.40)` | Default | `ln(1.40)` *(truncated >128px knives)* | **`ln(1.95)` (`0.6678` rad)** |
| **Optimizer & Weight Decay** | SGD (`lr=0.005`) | SGD / AdamW | AdamW (`1e-4`) | AdamW (`8e-5`) | AdamW (`5e-5`) | **AdamW (`wd=0.0005`, Cosine Annealing LR)** |
| **Base Learning Rate** | `0.005` | `0.0001` | `0.0001` | `0.00008` | `0.00005` | **M10–M11: `4.5e-5` · M12: `4.0e-5` · M13: `3.5e-5`** |
| **Batch Size & Gradient Clip** | Batch 4 | Batch 4 | Batch 2, Clip `5.0` | Batch 2, Clip `5.0` | Batch 2, Clip `5.0` | **Batch 2, Max Grad Norm `5.0`, FP16 AMP** |
| **Surveillance Augmentations** | Horizontal flip only | Flip + brightness | Flip + blur | Flip + specular jitter | Flip + blur + glint | **Flip (50%) + Motion Blur (`k=5,7`) + Blade Glint + Specular Jitter** |
| **WiSE-FT Weight Calibration** | None | None | None | None | None | **M12: $\alpha=0.30$ (with M11) · M13: $\alpha=0.50$ (with M12)** |

---

## 2. Dataset Provenance Audit (What Datasets Are Included vs. Excluded)

> **Critical Provenance Rule for Panel Defense:**  
> Early exploratory datasets used in Models 1–4 (**Open Images V6**, **ARMAS**, and **Hugging Face macro web scrapes**) were **100% purged from Model 5 onward**. They contain **0 images** in Models 5, 6, 7, 8, 9, 10, 11, 12, and 13.

| Dataset Source | Domain Type | Used in Which Models? | Included in Models 10–13? | Purpose & Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Open Images V6** | Web / studio photography | Models 1–2 only | **NO (0 images)** | Discarded after Model 2 due to severe domain/scale mismatch with CCTV. |
| **ARMAS Dataset** | Mixed web firearm images | Model 3 only | **NO (0 images)** | Discarded after Model 3 due to corrupted coordinate annotations in Batch 2. |
| **Hugging Face Scrape** | Close-up macro web images | Models 3–4 only | **NO (0 images)** | Discarded after Model 4; macro photos caused high false alarms on CCTV backgrounds. |
| **USRT Surveillance Dataset** | Real overhead CCTV footage | **Models 5–13** | **YES** | Core real-world overhead CCTV handgun and background negative corpus. |
| **OD-WeaponDetection** | Surveillance & indoor weapons | **Models 5–13** | **YES** | Core real-world knife and handgun detection corpus. |
| **VIRAT 2.0 Surveillance** | Unconstrained outdoor CCTV | **Models 6–13** | **YES** | Real surveillance hard-negative scenes (pedestrians, vehicles, bags, shadows). |
| **Knife.v1 Surveillance** | Multi-angle surveillance blades | **Models 8–13** | **YES** | Distant, angled, and motion-blurred knife blade instances. |
| **CostiCatargiu (`WeaponDataset v11` & `NoGUN v5`)** | Curated weapons + held everyday objects | **Models 10–13** | **YES** | `450` held non-weapon images (`NoGUN v5`) + `550` curated handgun/knife images (`WeaponDataset v11`). |
| **Staged Tapo `CAM 01` & `CAM 02` (Batch 1)** | Dual-camera indoor/outdoor CCTV | **Models 11–13** | **YES** | `390` train frames + `44` held-out val frames + `250` ROI crops of groupmates holding phones, wallets, bottles, umbrellas. |
| **Handheld Cup / Mug / Torso Crops (`v13_cups`)** | Targeted torso/cup confounders | **Models 12–13** | **YES** | `40` targeted handheld cup/mug/drink ROI crops extracted strictly from train negatives. |
| **Nov 20 Staged Household Non-Weapons (`Scenes #01–#06`)** | Dual-camera indoor household look-alikes | **Model 13 only** | **YES (Model 13)** | `240` full frames + `240` tight FP ROI crops (`v14`) from 12 non-weapon clips (black/long TV remotes, metal ladle, white rag, phone, wallet). Weapon prop Scenes `#07–#11` 100% excluded. |

---

## 3. Exact Dataset Split Counts & Class Distributions (Models 1 to 13)

### Table A: Full Training, Validation & Test Split Progression (Models 1 through 13)

| Model | Train Total Images | Train Handguns (Imgs / Boxes) | Train Knives (Imgs / Boxes) | Train Full-Frame Hard Negatives | Injected Negative ROI Crops | Effective Training Pool | Validation Split | Test Split |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | 681 | 341 | 340 | 0 | 0 | 681 | 85 | Legacy 86 |
| **Model 2** | 681 | 341 | 340 | 0 | 0 | 681 | 85 | Legacy 86 |
| **Model 3** | 1,596 | 798 | 798 | 0 | 0 | 1,596 | 342 | Legacy 343 |
| **Model 4** | 1,553 | 755 | 798 | 0 | 0 | 1,553 | 333 | Legacy 333 |
| **Model 5** | 1,653 | 751 | 752 | 150 | 0 | 1,653 | 354 | Legacy 356 |
| **Model 6** | 1,765 | 751 | 752 | 262 | 96 | 1,861 | 378 | Legacy 380 |
| **Model 7** | 1,838 | 751 | 752 | 335 | 180 | 2,018 | 393 | Legacy 397 |
| **Model 8** | 2,095 | 902 | 908 | 285 | 240 | 2,335 | 260 | Legacy 265 (`228 W / 37 Neg`) |
| **Model 9** | 2,214 | 902 | 908 | 404 | 296 | 2,510 | 275 | **279 (Frozen)** |
| **Model 10** | 2,804 | 962 (1,018 boxes) | 988 (1,092 boxes) | 854 | 296 | 3,100 | 275 | **279 (Frozen)** |
| **Model 11** | 4,793 | 1,850 (1,926 boxes) | 1,699 (1,829 boxes) | 1,244 | 546 | 5,339 | **650** | **279 (Frozen)** |
| **Model 12 (Video SOTA)** | **5,203** | **1,990 (2,071 boxes)** | **1,969 (2,118 boxes)** | **1,244** | **586** | **5,789** | **650** | **279 (Frozen)** |
| **Model 13 (Static/HN SOTA)** | **5,443** | **1,990 (2,071 boxes)** | **1,969 (2,118 boxes)** | **1,484** *(+240 Nov20, 2× sampled)* | **826** *(+240 Nov20 `v14`)* | **6,269** | **650** | **279 (Frozen)** |

---

### Table B: Detailed Breakdown of Injected Hard-Negative ROI Crop Folders (Models 12 & 13)

| Negative ROI Crop Folder (`research/model_improvement/dataset/`) | Crop Count | Target Confounder / False-Positive Source | Used in Model 12? | Used in Model 13? |
| :--- | :---: | :--- | :---: | :---: |
| `hard_negatives` | 96 | Early VIRAT & indoor background shadow/edge traps | Yes | Yes |
| `hard_negatives_fullframe` | 48 | Retail POS counter pinpads, cash registers, dining hall fixtures | Yes | Yes |
| `hard_negatives_v9` | 56 | Door frames, dark floor tiles, wall corners | Yes | Yes |
| `hard_negatives_v10` | 40 | Specular metal reflections & smartphone screen edges | Yes | Yes |
| `hard_negatives_v11` | 56 | Hand-shadow & bare wrist/forearm motion crops | Yes | Yes |
| `hard_negatives_v12_staged` | 250 | Staged Tapo `CAM 01`/`02` held phones, wallets, water bottles, umbrellas | Yes | Yes |
| `hard_negatives_v13_cups` | 40 | Upright handheld coffee mugs, soda cans, cups & mid-torso hand regions | Yes | Yes |
| `hard_negatives_v14_nov20_household` | **240** | Nov 20 `Scenes #01–#06` black/long TV remotes, metallic kitchen ladle, rolled white rag, phone, wallet | No | **Yes (+240)** |
| **Total Targeted Negative ROI Crops** | **586 (M12) / 826 (M13)** | — | **586** | **826** |

---

### Table C: Frozen Evaluation & Benchmark Partitions (Zero Data Leakage Guaranteed)

| Benchmark Partition | Total Images / Frames | Class 1 (`Handgun`) | Class 2 (`Knife`) | Class 0 (`Hard Negative`) | Overlap with Train / Val (Models 1–13) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Split (`val_ids.txt`)** | 650 images | 254 images (267 boxes) | 242 images (263 boxes) | 154 images | **0.0% overlap** with Train & all Test sets |
| **Internal Test Split (`Test 279`)** | 279 images | 114 images (114 boxes) | 114 images (124 boxes) | 51 images | **0.0% overlap** with Train & Val (M8–M13) |
| **Strictly Unseen Internal CCTV (`CCTV 185`)** | 185 images | 114 images | 31 images | 40 images | **0.0% overlap** across all 13 models |
| **Unified External Benchmark (`Ext 350`)** | 350 images | 134 images | 116 images | 100 images *(incl. 60 held confounders)* | **0.0% overlap** across all 13 models |
| **16-Stream Real CCTV Video GT** | 16 videos (2,538 frames) | 3 Orig/Enh pairs (`HG01–03`) | 3 Orig/Enh pairs (`KN01–03`) | 4 No-Weapon videos (`NW01–04`, 948 frames) | **0.0% overlap** across all 13 models |
