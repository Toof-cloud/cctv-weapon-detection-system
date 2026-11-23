# Thesis 2 Panel Recommendations & Proposed Response

## 1. Panel Concerns

**Concern 1 — High false negative rate.**
The panel questioned the high false negative rate of Model 9 (formerly Candidate V11). Although it has the highest precision of all models (**94.09%**, only **11 false positives**), its overall recall is only **73.53%**: **63 of 238 weapons (26.47%) were missed**. Because the system supports video analysis professionals through object detection, video enhancement, and report generation, every missed weapon is a critical failure. The false negative rate must be reduced.

**Concern 2 — Weapon look-alike objects.**
The system should not detect objects that only resemble weapons (e.g., a person holding a pen, an eggplant, or other handgun-/knife-shaped objects). The panel advised adding **negative samples** of people holding look-alike objects to the training data.

### Model 9 (V11) baseline

| Class | TP | FP | FN | Precision | Recall | FN rate | AP@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Handgun | 110 | 5 | 4 | 95.65% | 96.49% | 3.51% | 90.82% |
| Knife | 65 | 6 | **59** | 91.55% | 52.42% | **47.58%** | 53.24% |
| **Overall** | **175** | **11** | **63** | **94.09%** | **73.53%** | **26.47%** | **72.03% (mAP)** |

59 of the 63 misses (94%) are knives, so reducing false negatives is mostly about improving knife detection.

## 2. Proposed Response (for adviser approval)

### A. Staged positive dataset (our own CCTV)
- Record airsoft/toy handguns and realistic training/prop knives with our CCTV cameras.
- Vary distance (near/mid/far), camera angle, lighting (day/night/IR), motion (walking, drawing, swinging), occlusion, and grip.
- Prioritize **knives** (main FN source).
- Annotate with the existing `handgun` / `knife` classes.

### B. Hard negatives — online first
- Collect images of people holding look-alike objects from public datasets (e.g., Open Images: Pen, Mobile phone, Remote control, Flashlight, Spatula, Scissors, Hair dryer, Eggplant, Stapler, Banana).
- Train as background-only (no boxes), consistent with the existing hard-negative pipeline.

### C. Hard negatives — staged if not available online
- Record the same CCTV setups with group members holding pens, eggplants, phones, remotes, wallets, flashlights, spatulas, combs, umbrellas, empty hands, etc.
- Mirror the poses/motions used in the positive staged dataset so the model learns the object, not the gesture.

### D. Evaluation
- Report per-class recall / FN rate, precision, mAP@50, and a new **look-alike false positive rate** on a held-out look-alike test set.
- Proposed target: knife recall ≥ 80% with look-alike FP rate ≤ 5%.

## 3. Questions for the Adviser
1. Is a staged airsoft/toy/prop dataset acceptable as training data for the thesis?
2. Should staged data be train-only, or also used as a separate staged test set alongside the existing real-CCTV test set?
3. Is a staged hard-negative set acceptable when online images are insufficient?
4. Are there institutional requirements (consent forms, permission for replica weapons on campus, filming location)?

---

## 4. Model 10 Resolution & Empirical Audit Summary
To directly resolve **Concern 1 (Knife Recall Truncation)** and **Concern 2 (Everyday Confounders)**:
* **Dataset Expansion (3,488 Total Images | 2,804 Train / 405 Val / 279 Test):**
  * **Handguns (Class 1):** 962 train images (902 USRT CCTV + 60 CostiCatargiu WeaponDataset v11).
  * **Knives (Class 2):** 988 train images (908 OD-WeaponDetection + 80 CostiCatargiu WeaponDataset v11).
  * **Hard Negatives (Class 0):** 854 train images (160 VIRAT 2.0, 80 USRT No_Gun, 60 Dining Hall Clip 08, 48 Retail Counter POS Pinpads Clips 05/09/10, 40 Staged Rooms, 16 Floor Smartphones, 450 CostiCatargiu NoGUN v5).
* **Architectural Fix for Concern 1:**
  * Restored the full five-stage anchor pyramid `(16, 32, 64, 128, 256)` and implemented calibrated $\ln(2.2)$ bounding box clamping, eliminating the 128px proposal truncation that caused the 59 knife false negatives in Model 9.
  * **Held-Out Test Result:** Knife AP surged from **50.43%** to **95.18%** (Overall mAP: **95.22%**).
* **Unified 350-Image External Benchmark Audit (Zero Data Leakage):**
  * Achieved **62.38% mAP@0.50** (highest among Models 1–10), **72.05% Precision**, and **67.03% Recall**.
* **Resolution for Concern 2 (Look-Alike & Everyday Objects):**
  * **External Benchmark Hard Negatives:** 84/100 non-weapon scenes rejected (**84.00% specificity**).
  * **Everyday Held-Object Grid Audit:** 17/18 clean cells rejected (**94.44% specificity**).
  * **Real-World Unconstrained CCTV Video Testing:** **0 false alarms** across **1,517 frames** (5 unconstrained videos; **100% specificity**).
