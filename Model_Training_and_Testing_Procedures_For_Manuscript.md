# All Models (1 to 9) Training, Dataset, and Testing Technical Factsheet

* **Thesis Project:** FORENSIKADA (Multi-Camera CCTV Weapon Detection System)  
* **University:** National University – Manila (*CCIT – Computer Science Department*)  
* **Purpose:** Consolidated factual technical data for all model iterations (Models 1 through 9) for direct integration into thesis manuscript tables, methodology (Chapter 3), results & discussion (Chapter 4), and defense presentations.

---

## Section 1: Master Comparative Summary Table (Models 1 to 9)

| Model | Dataset & Sourcing Focus | Train / Val / Test Images | Precision | Recall | F1-Score | AP (Handgun) | AP (Knife) | mAP@0.50 | TP / FP / FN | Key Finding / Forensic Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 1** | Open Images V6 Subset (FiftyOne) | 681 / 85 / 86 (852 Total) | 65.94% | 71.09% | 68.42% | 75.99% | 65.16% | 70.58% | 91 / 47 / 37 | Baseline prototype. Limited sample volume; low knife recall in CCTV. |
| **Model 2** | Open Images V6 (Cleaned Redundancies) | 681 / 85 / 86 (852 Total) | 76.92% | 70.31% | 73.47% | 75.23% | 66.39% | 70.81% | 90 / 27 / 38 | Precision improved by 11% by removing bad boxes; dataset still too small. |
| **Model 3** | Armas + Hugging Face + Howard Knife | 1,596 / 342 / 343 (2,281 Total) | 48.14% | 58.20% | 52.69% | 09.69% | 83.65% | 46.67% | 220 / 237 / 158 | Handgun AP collapsed to 9.69% due to 207 bad scaling annotations in Batch 2. |
| **Model 4** | Sanitized Batch 2 (88 bad purged) | 1,553 / 333 / 333 (2,219 Total) | 67.06% | 59.95% | 63.31% | 12.76% | 91.33% | 52.04% | 226 / 111 / 151 | Knife reached 91.3%, but handguns failed in CCTV (0% TP; macro studio scale mismatch). |
| **Model 5** | USRT CCTV-Gun (Mendeley) + Knife + Negs | 1,653 / 354 / 356 (2,363 Total) | 85.37% | 90.93% | 88.07% | 85.97% | 89.78% | 87.87% | 321 / 55 / 32 | Handgun crisis solved! AP surged to 85.97%. Fan & door false alarms eliminated. |
| **Model 6** | VIRAT CCTV Negatives + Augmented Knife | 1,765 / 378 / 380 (2,523 Total) | 75.17% | 93.48% | 83.33% | 91.77% | 92.68% | 92.23% | 330 / 109 / 23 | Highest Recall (93.48%). CCTV knife sensitivity surged to 100 frames. Over-triggers on diagonal textures. |
| **Model 7** | Hard Negative Mining (Bare Hands/Rails) + Small Anchors + Cosine LR | 1,838 / 393 / 397 (2,628 Total) | 70.48% | 90.65% | 79.31% | 86.98% | 88.10% | 87.54% | 320 / 134 / 33 | 85.7% hard negative rejection. Small anchors ((8, 16, 32, 64, 128)) triggered on sub-12px noise. |
| **Model 8** | Balanced Anchors (16–256px) + Sanitized Negatives (Phones/VIRAT) + Specular Jitter | 2,095 / 260 / 265 (2,620 Total) | 81.18% | 92.44% | 86.44% | 94.02% | 88.22% | 91.12% | 220 / 51 / 18 | Silver gun flaw solved (AP surged to 94.02%). Phone floor false alarms eliminated. |
| **Model 9** | Surveillance Hard Negatives (Dining/POS Pinpads/Bottles) + 1D Directional Motion Blur (PSF) | 2,214 / 275 / 279 (2,768 Total) | **88.64%** | **93.30%** | **90.90%** | **93.59%** | **91.91%** | **92.75%** | **222 / 29 / 16** | **Highest mAP (92.75%) in project history!** Hard negative rejection reached 92.16%. POS pinpad false alarms 100% eliminated; bottles slashed by 83.3%. |

---

## Section 2: Detailed Model-by-Model Factsheets

### Model 1: Baseline Open Images Prototype
* **Checkpoint Filename:** `best_weapon_detector.pth`
* **Dataset Name:** Open Images V6 Weapon Subset
* **Sourced From:** FiftyOne / Google Open Images
* **Target Classes:** Class 1: Handgun | Class 2: Knife
* **Dataset Statistics:**
  * Total Images: 852
  * Total Annotations: 1,114 bounding boxes
  * Handgun Class Total: 268 images
  * Knife Class Total: 584 images
* **Split Configuration (80% Train / 10% Val / 10% Test):**
  * Training Set (80%): 681 images (219 Handgun, 462 Knife)
  * Validation Set (10%): 85 images (29 Handgun, 56 Knife)
  * Test Set (10%): 86 images (20 Handgun, 66 Knife)
* **Training Hyperparameters:**
  * Architecture: Faster R-CNN (ResNet-50 FPN)
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Batch Size: 2 | Epochs: 10
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **65.94%** (0.659420)
  * Recall: **71.09%** (0.710938)
  * F1-Score: **68.42%** (0.684211)
  * AP (Handgun): **75.99%** (0.759953)
  * AP (Knife): **65.16%** (0.651640)
  * mAP@0.50: **70.58%** (0.705797)
  * True Positives: 91 | False Positives: 47 | False Negatives: 37
* **Forensic Assessment:**
  * Baseline feasibility established, but dataset volume (852 images) was too small.
  * Knife detection struggled significantly in low-resolution video conditions.

---

### Model 2: Cleaned Open Images Dataset
* **Checkpoint Filename:** `best_weapon_detector_retrained.pth`
* **Dataset Name:** Open Images V6 Weapon Subset (Cleaned)
* **Sourced From:** FiftyOne / Open Images (curated)
* **Preprocessing:** Removed redundant, duplicate, and misaligned boxes from the Open Images set.
* **Dataset Statistics:** Total Images: 852 | Total Annotations: 1,114 bounding boxes
* **Split Configuration (80% Train / 10% Val / 10% Test):**
  * Training Set (80%): 681 images (219 Handgun, 462 Knife)
  * Validation Set (10%): 85 images (29 Handgun, 56 Knife)
  * Test Set (10%): 86 images (20 Handgun, 66 Knife)
* **Training Hyperparameters:**
  * Architecture: Faster R-CNN (ResNet-50 FPN)
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Batch Size: 2 | Epochs: 10
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **76.92%** (0.769231) *(+10.98% precision surge)*
  * Recall: **70.31%** (0.703125)
  * F1-Score: **73.47%** (0.734694)
  * AP (Handgun): **75.23%** (0.752293)
  * AP (Knife): **66.39%** (0.663856)
  * mAP@0.50: **70.81%** (0.708075)
  * True Positives: 90 | False Positives: 27 *(dropped from 47 to 27)* | False Negatives: 38
* **Forensic Assessment:**
  * Cleaned annotations reduced false positive bounding boxes.
  * Still lacked diverse CCTV angles and camera distance variations.

---

### Model 3: Expanded Multi-Source Dataset
* **Checkpoint Filename:** `best_weapon_detector_third_model.pth`
* **Dataset Sources:**
  * Handgun: DATASET- ARMAS COMPUTER VISION MODEL (Roboflow) + WEAPON DETECTION DATASET (Hugging Face / Dataset Ninja)
  * Knife: KNIFE DATASET - OD-WEAPONDETECTION (University of Granada / Dataset Ninja, curated by Howard)
* **Storage Folder:** `training/model_3/third_model_data/`
* **Dataset Statistics:** Total Images: 2,281 | Total Annotations: 2,522 bounding boxes (Handgun: 1,146 imgs / 1,322 boxes; Knife: 1,135 imgs / 1,200 boxes)
* **Split Configuration (70% Train / 15% Val / 15% Test):**
  * Training Set (70%): 1,596 images (802 Handgun / 913 boxes, 794 Knife / 843 boxes; 1,756 annos)
  * Validation Set (15%): 342 images (172 Handgun / 208 boxes, 170 Knife / 180 boxes; 388 annos)
  * Test Set (15%): 343 images (172 Handgun / 201 boxes, 171 Knife / 177 boxes; 378 annos)
* **Training Hyperparameters:**
  * Architecture: Faster R-CNN (ResNet-50 FPN v2)
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Batch Size: 2 | Epochs: 10
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **48.14%** (0.481400)
  * Recall: **58.20%** (0.582011)
  * F1-Score: **52.69%** (0.526946)
  * AP (Handgun): **09.69%** (0.096947) *(SEVERE COLLAPSE)*
  * AP (Knife): **83.65%** (0.836480)
  * mAP@0.50: **46.67%** (0.466714)
  * True Positives: 220 | False Positives: 237 | False Negatives: 158
* **Forensic Diagnosis & Defect Identification:**
  * Detailed inspection revealed that 207 images in Jabez's Handgun Batch 2 contained invalid coordinates:
    * 88 flat-line (zero width/height) and out-of-bounds bounding boxes.
    * 119 coordinate normalization scaling errors (bounding boxes shifted off the weapon).
  * Corrupted Faster R-CNN regression loss and collapsed handgun detection.

---

### Model 4: Sanitized & Balanced Dataset
* **Checkpoint Filename:** `best_weapon_detector_fourth_model.pth`
* **Dataset Sources:**
  * Handgun: DATASET- ARMAS (Roboflow) + Sanitized Hugging Face (Jabez Batch 2 with 88 corrupt boxes purged)
  * Knife: KNIFE DATASET - OD-WEAPONDETECTION (Howard clean knife set)
* **Storage Folder:** `training/model_4/fourth_model_data/`
* **Dataset Statistics:** Total Images: 2,219 | Total Annotations: 2,434 boxes (Handgun: 1,084 imgs / 1,234 boxes; Knife: 1,135 imgs / 1,200 boxes; 1:1 balanced distribution)
* **Split Configuration (70% Train / 15% Val / 15% Test):**
  * Training Set (70%): 1,553 images (759 Handgun / 856 boxes, 794 Knife / 843 boxes; 1,699 annos)
  * Validation Set (15%): 333 images (163 Handgun / 190 boxes, 170 Knife / 180 boxes; 370 annos)
  * Test Set (15%): 333 images (162 Handgun / 188 boxes, 171 Knife / 177 boxes; 365 annos)
* **Training Hyperparameters:**
  * Architecture: Faster R-CNN (ResNet-50 FPN v2)
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Batch Size: 2 | Epochs: 10
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **67.06%** (0.670623)
  * Recall: **59.95%** (0.599469)
  * F1-Score: **63.31%** (0.633053)
  * AP (Handgun): **12.76%** (0.127583) *(Handgun AP remained critically low)*
  * AP (Knife): **91.33%** (0.913288) *(Knife achieved high accuracy)*
  * mAP@0.50: **52.04%** (0.520436)
  * True Positives: 226 | False Positives: 111 | False Negatives: 151
* **Video Benchmark Failure Analysis (The Handgun Crisis):**
  * In `samples/CAM02_Scene_004.mp4`: 0 true handgun detections. Model bounded the black electric fan 53 times (78.7% conf) and misclassified the actual handgun as a knife (93.4% conf).
  * In `samples/handgun_test-video.mp4`: 0 true handgun detections. Model repeatedly bounded the frosted glass window transom above the doorway (94.9% conf).
* **Root Cause (Scale & Domain Mismatch):**
  * Jabez's handgun training images consisted of studio catalog photos where the gun occupied 40% to 80% of the image frame.
  * In authentic CCTV surveillance footage, a handgun in an actor's hand occupies only ~0.35% of the frame.
  * Faster R-CNN anchor generators trained on macro shots could not resolve tiny surveillance handguns.

---

### Model 5: USRT Real-Time CCTV Handgun + Clean Knife + Negatives
* **Checkpoint Filename:** `best_weapon_detector_fifth_model.pth` *(Size: 173.4 MB)*
* **Dataset Sources:**
  * Handgun: Action recognition and object detection dataset for firearm-related actions (*Ruiz-Santaquiteria et al., 2023, Mendeley Data DOI: 10.17632/bbzpxhd22j.2*)
  * Knife: OD-WeaponDetection: Knife Detection (*Howard clean knife set*)
  * Negatives: Action recognition firearm dataset (*No_Gun non-weapon activity subset, Mendeley Data*)
* **Storage Folder:** `training/model_5/fifth_model_data/`
* **Dataset Statistics:** Total Images: 2,363 | Total Annotations: 2,328 boxes (Handgun: 1,128 imgs / 1,128 boxes; Knife: 1,135 imgs / 1,200 boxes; CCTV Negatives: 100 images; Mean box area = 0.35% of frame)
* **Split Configuration (70% Train / 15% Val / 15% Test):**
  * Training Set (70%): 1,653 images (789 Handgun, 794 Knife, 70 Negatives; 1,632 annos)
  * Validation Set (15%): 354 images (169 Handgun, 170 Knife, 15 Negatives; 349 annos)
  * Test Set (15%): 356 images (170 Handgun, 171 Knife, 15 Negatives; 347 annos)
* **Training Hyperparameters:**
  * Architecture: Faster R-CNN (ResNet-50 FPN v2)
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Batch Size: 2 | Epochs: 10
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **85.37%** (0.853723)
  * Recall: **90.93%** (0.909348)
  * F1-Score: **88.07%** (0.880658)
  * AP (Handgun): **85.97%** (0.859718) *(Surged by +73.21% over Model 4!)*
  * AP (Knife): **89.78%** (0.897765)
  * mAP@0.50: **87.87%** (0.878741) *(Surged by +35.83% over Model 4!)*
  * True Positives: 321 | False Positives: 55 | False Negatives: 32
* **Forensic Video Breakthroughs:**
  * In `samples/handgun_test-video.mp4`: 94–95 True Handgun Detections (mean conf: 94.8%). Doorway/frosted glass false alarms = 0.
  * In `samples/CAM02_Scene_004.mp4`: 49–51 True Handgun Detections (conf up to 96.84%). Standing electric fan false alarm = 0 frames (100% eliminated).
* **Remaining Limitations:**
  * Knife sensitivity in low-res CCTV was weak (detected in only 3 frames in `NEW_KNIFE_VIDEO`).
  * Background negative coverage needed multi-camera outdoor public surveillance context (VIRAT).

---

### Model 6: VIRAT CCTV Negatives + Surveillance-Augmented Knives
* **Checkpoint Filename:** `best_weapon_detector_sixth_model.pth` *(Size: 173.4 MB)*
* **Dataset Sources:**
  * Handgun: USRT Action recognition firearm dataset (*Mendeley Data DOI: 10.17632/bbzpxhd22j.2*)
  * Knife: OD-WeaponDetection: Knife Detection (Howard Clean) + Dynamic CCTV Domain Augmentations
  * CCTV Negatives: VIRAT Video Dataset Release 2.0 (Kitware Data / DARPA, 160 frames) + USRT No_Gun (100 frames)
* **Storage Folder:** `training/model_6/sixth_model_data/`
* **Dataset Statistics:** Total Images: 2,523 | Total Annotations: 2,328 boxes (Handgun: 1,128; Knife: 1,135; CCTV Negatives: 260 [10.3% of total data])
* **Split Configuration (70% Train / 15% Val / 15% Test):**
  * Training Set (70%): 1,765 images (789 Handgun, 794 Knife, 182 CCTV Negatives; 1,632 annos)
  * Validation Set (15%): 378 images (169 Handgun, 170 Knife, 39 CCTV Negatives; 349 annos)
  * Test Set (15%): 380 images (170 Handgun, 171 Knife, 39 CCTV Negatives; 347 annos, 353 targets)
* **Training Hyperparameters:**
  * Architecture: Faster R-CNN (ResNet-50 FPN v2)
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Batch Size: 2 | Epochs: 10 (8,830 total batch updates)
  * Hardware: NVIDIA GeForce RTX 5060 Ti GPU (16 GB VRAM)
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **75.17%** (0.751708)
  * Recall: **93.48%** (0.934844) *(HIGHEST RECALL: Only 23 missed targets out of 353!)*
  * F1-Score: **83.33%** (0.833333)
  * AP (Handgun): **91.77%** (0.917726)
  * AP (Knife): **92.68%** (0.926775)
  * mAP@0.50: **92.23%** (0.922251)
  * True Positives: 330 | False Positives: 109 | False Negatives: 23
* **4-Quadrant Video Benchmark Results:**
  * Handgun Real CCTV (`handgun_test-video.mp4`): 132 handgun detections (0 knife cross-detections; 93.55% avg conf).
  * Knife Real CCTV (`NEW_KNIFE_VIDEO_11s.mp4`): 100 knife detections (jumped from 3 in Model 5; peak conf 93.81%).
  * Handgun Staged CCTV (`CAM02_Scene_004.mp4`): 158 handgun detections tracking suspect down stairs (peak conf 88.5%).
  * Knife Evaluation Video (`evaluation_video.mp4`): 42 knife detections (peak conf 94.7%).

---

### Model 7: Hard Negative Mining + Small Anchors + Cosine Annealing
* **Checkpoint Filename:** `best_weapon_detector_seventh_model.pth`
* **Dataset Sources:**
  * Handgun: USRT Firearm Action Recognition Dataset
  * Knife: OD-WeaponDetection (Howard Clean) + CCTV Domain Augmentations + Synthetic Infrared / Night-Vision Phosphor Mode (p=0.25)
  * Authentic CCTV Hard Negatives: Bare hands, clenched fists, empty-handed pedestrians (USRT No_Gun) + Architectural railings, stairs, indoor structures (VIRAT Release 2.0). Total: 365 images (13.9% of data).
* **Storage Folder:** `training/model_7/seventh_model_data/`
* **Dataset Statistics:** Total Images: 2,628 | Total Annotations: 2,328 boxes (Handgun: 1,128; Knife: 1,135; Negatives: 365)
* **Split Configuration (70% Train / 15% Val / 15% Test):**
  * Training Set (70%): 1,838 images (789 Handgun, 794 Knife, 255 Hard Negatives; 1,632 annos)
  * Validation Set (15%): 393 images (169 Handgun, 170 Knife, 54 Hard Negatives; 349 annos)
  * Test Set (15%): 397 images (170 Handgun, 171 Knife, 56 Hard Negatives; 347 annos, 353 targets)
* **Training Hyperparameters & Pipeline Enhancements:**
  * Architecture: Faster R-CNN (ResNet-50 FPN v2)
  * Small-Scale RPN Anchors: `sizes=((8,), (16,), (32,), (64,), (128,))`, `aspect_ratios=((0.5, 1.0, 2.0),) * 5`
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Learning Rate Schedule: Cosine Annealing LR (`T_max=15 epochs`, `eta_min=1e-5`)
  * Thread Capping: `--threads 8`, `torch.set_num_threads(8)`, `OMP_NUM_THREADS=8`
  * Batch Size: 2 | Epochs: 15 (13,785 total batch updates)
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **70.48%** (0.704846)
  * Recall: **90.65%** (0.906516)
  * F1-Score: **79.31%** (0.793061)
  * AP (Handgun): **86.98%** (0.869788)
  * AP (Knife): **88.10%** (0.881026)
  * mAP@0.50: **87.54%** (0.875407)
  * True Positives: 320 | False Positives: 134 | False Negatives: 33
  * Hard Negative Rejection Specificity: **48/56 images (85.71% complete zero-alarm rate)**
* **Diagnosis:**
  * Small 8px anchors hyper-sensitized the detector to sub-12px noise (door hinges, switch plates, bolts).
  * 25 CAM02 negative frames accidentally penalized the actor's silver firearm.

---

### Model 8: Balanced Anchor CCTV Detector & Distractor Sanitization
* **Checkpoint Filename:** `best_weapon_detector_eighth_model.pth`
* **Dataset Name:** CCTV Multi-Source Sanitized Dataset with Balanced Anchors
* **Sourced From:**
  * Handgun (Class 1): USRT CCTV-Gun Dataset (1,128 surveillance images)
  * Knife (Class 2): Howard CCTV Augmented Dataset (1,135 clean images)
  * Hard Negatives (Class 0): 357 curated negative frames (200 VIRAT streets/stairs, 100 USRT No_Gun, 40 clean rooms, 17 horizontal victim floor phones; purged 25 contaminated CAM02 frames).
* **Storage Folder:** `training/model_8/eighth_model_data/`
* **Dataset Statistics:** Total Images: 2,620 | Total Annotations: 2,685 records
* **Split Configuration (80% Train / 10% Val / 10% Test):**
  * Training Set (80%): 2,095 images (902 Handgun, 908 Knife, 285 Negatives; 2,150 annos)
  * Validation Set (10%): 260 images (113 Handgun, 113 Knife, 34 Negatives; 265 annos)
  * Test Set (10%): 265 images (113 Handgun, 114 Knife, 38 Negatives; 270 annos; 37 pure zero-box negatives)
* **Architecture & Balanced Anchor Geometry:**
  * Architecture: Faster R-CNN (ResNet-50 FPN v2)
  * Anchor Sizes: Balanced Scales `((16,), (32,), (64,), (128,), (256,))`
  * Aspect Ratios: `((0.5, 1.0, 2.0),) * 5`
  * Scale Invariance: Covers distant weapons (16–32px) and brandished firearms (64–128px) while terminating sub-12px noise triggers.
* **CCTV Domain Augmentation Pipeline:**
  * Specular Highlight & Metal Tone Jitter: $\pm 25\%$ brightness/contrast on firearms (chrome/silver & matte black).
  * Lens & Motion Blur: Gaussian blur (radius 1.0 – 2.0).
  * Resolution Downsampling: Bilinear scale reduction (0.5x – 0.75x) simulating 480p/720p surveillance streams.
  * Synthetic Infrared (IR) Mode: Monochromatic sensor simulation with phosphor tint (20% probability).
* **Training Hyperparameters:**
  * Optimizer: SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Learning Rate Schedule: Cosine Annealing (`lr: 0.005 -> 1e-5`)
  * Batch Size: 2 | Epochs: 12 (Duration: 93m 54s)
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$):**
  * Precision: **81.18%** (0.811808) *(+10.70% over Model 7)*
  * Recall: **92.44%** (0.924370) *(+1.79% over Model 7)*
  * F1-Score: **86.44%** (0.864440) *(+7.13% over Model 7)*
  * AP (Handgun): **94.02%** (0.940203) *(+7.04% surge over Model 7)*
  * AP (Knife): **88.22%** (0.882200)
  * mAP@0.50: **91.12%** (0.911201) *(+3.58% over Model 7)*
  * True Positives: 220 | False Positives: 51 *(slashed by -61.9%)* | False Negatives: 18 *(dropped by -45.5%)*
  * Negative Image Rejection Rate: **31/37 (83.78% zero-alarm rate)**
* **4-Video Benchmark Validation Audit:**
  * `CAM02_Scene_004.mp4`: Flawless silver gun tracking up to 90.00% conf. Fan & door false alarms = 0.
  * `NEW_KNIFE_VIDEO_11s.mp4`: Knife tracked up to 90.33% conf; floor phone false alarm = 0.
  * `evaluation_video.mp4`: 45 confirmed alerts tracking knife up to 97.13% conf.
  * `handgun_test-video.mp4`: Handgun localized up to 93.93% conf; 0 door-hinge alarms.

---

### Model 9 (State-of-the-Art): Surveillance Hard Negatives & 1D Directional Motion Blur
* **Checkpoint Filename:** `best_weapon_detector_ninth_model.pth` *(Size: 329.6 MB)*
* **Dataset Name:** Real Surveillance Multi-Source Hard Negative & Motion-Blurred Dataset
* **Sourced From:**
  * Handgun (Class 1): USRT CCTV-Gun Dataset (1,128 authentic surveillance images across 140 scenes)
  * Knife (Class 2): Howard CCTV Augmented Dataset (1,135 clean images) + 1D Directional Motion Blur (PSF)
  * Hard Negatives (Class 0): 505 curated authentic surveillance negative frames:
    * 200 authentic VIRAT Public Surveillance multi-camera frames (parking, stairs, walkways)
    * 100 USRT No_Gun authentic CCTV walking pedestrian frames
    * 60 authentic indoor dining hall frames (Clip 08: beverage bottles, cups, trays, smartphones, wallets)
    * 48 pre-incident retail store counter frames (Clips 05, 09, 10: card payment terminals/POS pinpads, cash drawers, counter glass)
    * 40 staged indoor environment negative frames (CAM01/CAM02 Scene 001)
    * 40 clean negative room surveillance frames (tables, chairs, lamps)
    * 17 horizontal black smartphone/gadget filming frames (restaurant floor recording)
* **Storage Folder:** `training/model_9/ninth_model_data/`
* **Dataset Statistics:** Total Images: 2,768 images | Total Annotation Records: 2,833 records (Handgun: 1,128; Knife: 1,200; Class 0 Hard Negatives: 505 [18.2% of total data])
* **Split Configuration (80% Train / 10% Val / 10% Test):**
  * Training Set (80%): 2,214 images (902 Handgun, 908 Knife, 404 Negatives; 2,266 annos)
  * Validation Set (10%): 275 images (113 Handgun, 113 Knife, 49 Negatives; 282 annos)
  * Test Set (10%): 279 images (113 Handgun, 114 Knife, 52 Negatives; 285 annos; 51 zero-box negatives)
* **Architecture & Balanced Anchor Configuration:**
  * Architecture: Faster R-CNN (ResNet-50 FPN v2)
  * Anchor Sizes: Balanced Scales `((16,), (32,), (64,), (128,), (256,))`
  * Aspect Ratios: `((0.5, 1.0, 2.0),) * 5`
  * Multi-Scale Pyramid: Multi-scale Feature Pyramid Network (P2 to P6) with ROI Align
* **Advanced CCTV Augmentation Pipeline:**
  * 1D Directional Motion Blur: Linear Point Spread Function (PSF) kernel (size 5 to 11 px, angle 0 to 180 deg) simulating authentic surveillance motion blur during rapid weapon draws and knife lunges.
  * Specular Highlight & Metallic Jitter: $\pm 25\%$ brightness/contrast jitter simulating chrome, stainless, and matte finishes.
  * CCTV Downsampling: Bilinear scale compression (0.5x to 0.75x) simulating 480p/720p surveillance streams.
  * Synthetic Infrared (IR) / Night-Vision Mode: Phosphor noise simulation for night cameras (15% probability).
* **Training Hyperparameters:**
  * Optimizer: SGD (`Initial lr=0.005`, `momentum=0.9`, `weight_decay=0.0005`)
  * Learning Rate Schedule: Cosine Annealing (`lr: 0.005 -> 1e-5`)
  * Batch Size: 2 | Epochs: 12 (13,284 total batch updates)
  * Hardware: NVIDIA GeForce RTX 5060 Ti GPU (16 GB GDDR6 VRAM, CUDA 13.0, PyTorch 2.13.0)
* **Test Split Metrics ($\text{IoU} \ge 0.50$, $\text{Conf} \ge 0.50$ across 279 held-out test images):**
  * Precision: **88.64%** (0.886400) *(+7.46% improvement over Model 8!)*
  * Recall: **93.30%** (0.933000) *(+0.86% improvement over Model 8!)*
  * F1-Score: **90.90%** (0.909000) *(+4.46% improvement over Model 8!)*
  * AP (Handgun): **93.59%** (0.935890)
  * AP (Knife): **91.91%** (0.919100) *(+3.69% surge over Model 8!)*
  * mAP@0.50: **92.75%** (0.927500) *(HIGHEST OVERALL mAP IN PROJECT HISTORY: +1.63% over Model 8!)*
  * True Positives: **222** (107 Handgun, 115 Knife)
  * False Positives: **29** (10 Handgun, 19 Knife) *(False positives slashed from 51 to 29: -43.1%!)*
  * False Negatives: **16** (7 Handgun, 9 Knife) *(Missed detections dropped to only 16!)*
  * Hard Negative Scene Rejection Specificity: **47/51 images (92.16% complete zero-alarm rate on complex backgrounds)**
* **Key Empirical Breakthroughs:**
  1. **Card Payment Pinpad Hallucination:** 100% eliminated in real surveillance (`clip_09`: 0 knife frames, 30 pure handgun).
  2. **Glass Beverage Bottle Hallucination:** Slashed by 83.3% in real surveillance (`clip_10`: dropped from 18 to 3 frames).
  3. **Directional Motion Blur:** Successfully preserved knife detection under rapid lunge dynamics.
  4. **Best overall balance:** 93.59% Handgun AP, 91.91% Knife AP, and 92.16% background rejection rate.

---

## Section 3: System Hardware & Software Specifications (Table 4 / Chapter 3)

| Specification Component | System Configuration | Technical Details & Defense Note |
| :--- | :--- | :--- |
| **Graphics Processing Unit (GPU)** | NVIDIA GeForce RTX 5060 Ti | 16 GB GDDR6 VRAM, CUDA Compute 12.x/13.0 capability |
| **Host Processor (CPU)** | Ryzen 5 5600x | Concurrency capped at 8 worker threads (`OMP_NUM_THREADS=8`) |
| **System Memory (RAM)** | 16 GB DDR4 | Supports dual simultaneous high-resolution video streams |
| **Storage Subsystem** | NVMe M.2 Solid State Drive | Read/Write throughput > 3,500 MB/s for frame IO |
| **Operating System** | Windows 10 Pro | Primary development and testing workstation OS |
| **Python Interpreter** | Python 3.11.6 | Isolated via virtual environment (`.venv`) |
| **Deep Learning Framework** | PyTorch 2.13.0+cu130 | CUDA 13.0 acceleration with Tensor Core acceleration |
| **Computer Vision Engine** | OpenCV 4.14.0 (`cv2`) | Hardware-accelerated video decoding via FFmpeg backend |
| **Super-Resolution Engine** | BasicVSR++ Recurrent Network | Optical flow-guided video restoration for degraded feeds |

---

## Section 4: Video Input Requirements of the System

* **Reference Standard:** Philippine CCTV Technical Reference (*Resolution: 720p / $1280 \times 720$ @ 30 FPS*)
* **System Input Component:** OpenCV VideoCapture (`cv2`) — decodes any codec/container supported by the host system's FFmpeg backend.

### Video Requirements Matrix

| Requirement | Minimum (Practical Reference) | Maximum | Technical Defense / Justification |
| :--- | :--- | :--- | :--- |
| **Resolution** | 720p ($1280 \times 720$ pixels) | No fixed maximum | 720p is the floor reference aligned with the Philippine CCTV technical standard. Higher resolution preserves visual information for distant and small objects. Resolution alone does not guarantee detection; object visibility (distance, angle, lighting) determines actual performance. |
| **Frame Rate (FPS)** | 30 FPS (practical reference) | No fixed maximum | 30 FPS is the baseline supported by the Philippine CCTV technical reference. Higher FPS increases computational workload. The system's `frame_interval` parameter controls how many frames are submitted to the detector, decoupling throughput from source FPS. |
| **Video Format / Container** | Any format decodable by OpenCV (`cv2`) | No fixed maximum | Supports standard surveillance containers and codecs: MP4, AVI, MKV, MOV, TS (H.264, H.265/HEVC, MJPEG, MPEG-4). The binding condition is successful frame extraction. |
| **Video Duration** | No fixed minimum | No fixed maximum | Treated as a computational workload factor, not a correctness constraint. Long recordings can be processed in segments while preserving frame numbers and timestamps across forensic logs. |
| **File Size** | No fixed minimum | No fixed maximum | Bound only by available storage and host memory. No fixed file size threshold is imposed by the detection pipeline. |
| **Compression (Codec Quality)** | Decodable by OpenCV without frame read failure | No fixed maximum | Excessive compression introduces macroblocking and ringing artifacts. Upstream BasicVSR++ super-resolution mitigates compression artifacts before detection inference. |
| **Object Visibility** | Object must be sufficiently visible | N/A | Governed by distance, ambient lighting, motion blur, and occlusion. Completely occluded or pitch-black objects cannot be detected by any computer vision system. |

### Technical Notes for Manuscript (Chapter 3 – System Design / Constraints)
1. **Standard Alignment:** The 720p / 30 FPS reference is grounded in the Philippine CCTV Technical Reference Standard, establishing an empirically supported baseline for domestic surveillance system integration.
2. **Graceful Degradation:** The system does not reject videos below 720p; lower-resolution inputs can still be processed, but detection performance degrades proportionally with the loss of visual information.
3. **Temporal Decoupling:** Handled by the configurable `frame_interval` parameter (default: analyzes ~5 to 10 FPS regardless of raw source FPS), decoupling detector workload from hardware capture rates.
4. **Super-Resolution Upstream:** BasicVSR++ recurrent video super-resolution is applied upstream of Faster R-CNN to restore visual information degraded by compression and low resolution.
5. **Forensic Record Generation:** The system complies with the thesis requirements for Howard (UI) and Tyrone (Prototype) by exporting structured and traceable detection records conforming to 8 standard fields.
6. **Multi-Camera Integration:** Dual-stream ingestion supporting CAM-01 and CAM-02 with isolated spatial tracking and chronological timeline reconciliation.
7. **Temporal Consistency Filtering:** Suppresses single-frame transient false proposals (`SUPPRESSED_TEMPORAL_FLICKER`) and smooths target class labels using tracklet majority voting.

---

## Section 5: Forensic Record Generation & Multi-Camera Architecture

### A. Forensic-Oriented Detection Record Specification (Howard & Tyrone Compliance)
Every detection record produced by FORENSIKADA contains the following 8 standard fields:
1. `source_video` *(str)*: Exact filename of the surveillance video feed (e.g., `"CAM02_Scene_004.mp4"`).
2. `frame_number` *(int)*: Exact zero-indexed integer index of the analyzed video frame.
3. `timestamp_seconds` *(float)*: Timeline offset in seconds from the start of the video (formatted as `HH:MM:SS.mmm`).
4. `camera_id` *(str)*: Surveillance camera identifier (e.g., `"CAM-01"`, `"CAM-02"`).
5. `object_label` *(str)*: Classified weapon category (`"handgun"` or `"knife"`).
6. `bounding_box` *(list)*: Pixel coordinates in standard `[x1, y1, x2, y2]` format.
7. `confidence_score` *(float)*: Detection confidence score between 0.0 and 1.0 (e.g., `0.8742`).
8. `validation_status` *(str)*: Status flag: `"CONFIRMED_ALERT"` (actionable threat) or `"SUPPRESSED"` (with reason: `STATIC_BACKGROUND_TRAP`, `NO_PERSON_PROXIMITY`, `GEOMETRIC_OVERSIZED`, `SUPPRESSED_TEMPORAL_FLICKER`).

### B. Multi-Camera Integration Architecture
1. Accepts two simultaneous camera streams (e.g., CAM-01 Entrance, CAM-02 Hallway).
2. Runs independent spatial tracking state machines (`CentroidMotionTracker`) per camera ID to prevent cross-camera coordinate pollution.
3. Interleaves all detected incidents into a single, unified chronological timeline sorted strictly by `timestamp_seconds`.
4. Generates evidentiary crop images organized into per-camera subdirectories:
   * `outputs/.../crops/CAM-01/`
   * `outputs/.../crops/CAM-02/`

### C. Temporal Consistency Filter Specification
1. **Tracklet Matching:** Associating frame-level bounding boxes across time using spatial centroid Euclidean distance gating (`max_distance = 60 px`).
2. **Flicker Suppression:** Suppresses transient detections that occur for fewer than 2 consecutive frames (`min_hits = 2`). Detections failing this condition are marked `SUPPRESSED_TEMPORAL_FLICKER`.
3. **Majority Voting:** Reconciles label identity within a sustained tracklet to prevent rapid weapon classification hopping between handgun and knife.
4. **End-of-Video Reconciliation:** Retroactively elevates sustained multi-frame detections once a tracklet crosses the required persistence threshold.

---

## Section 6: Post-Detector Surveillance Intelligence & Anthropometric Validation

### A. Theoretical Foundation: Overcoming 2D Convolutional Scale Amnesia
Standard 2D Convolutional Neural Networks (including Faster R-CNN) operate on local pixel gradients without intrinsic 3D depth, physical scale, or anatomical awareness. Consequently, distant background fixtures (staircase stringers, table borders, door molding, backpack straps) frequently produce high-confidence weapon proposals when their 2D silhouettes match weapon aspect ratios.

Rather than indefinitely retraining network weights on an intractable variety of environmental textures, FORENSIKADA integrates a post-detector Surveillance Intelligence Layer enforcing:
1. **Kinematic Motion Tracking (`CentroidMotionTracker`):** Stationary objects with displacement $< 30\text{ px}$ over $\ge 10\text{ frames}$ are classified as `STATIC_BACKGROUND_TRAP`.
2. **Anatomical Reach Envelope:** Valid handheld weapons must fall within human arm manipulation range (`rel_y` in $[0.20, 0.80]$ relative to upright torso height; relaxed to $1.15$ when person is bottom-truncated by camera edge).
3. **Anthropometric Scale Constraints:**
   * Area Ratio: Weapon area $\le 20\%$ of associated person bounding box area.
   * Width Ratio: Weapon width $\le 75\%$ of person torso width.
   * Height Ratio: Weapon height $\le 45\%$ of effective person dimension ($\max(p_h, p_w)$).
4. **Aspect-Ratio Smartphone Gating:** Objects with $\text{aspect\_ratio} < 0.50$ (height $> 2\times$ width) associated with hand locations are recognized as vertical smartphones and suppressed.

### B. Empirical Stress-Test Results on Unconstrained Surveillance Footage

| Surveillance Video Clip | Scene Description / Environmental Challenge | Raw Detector Output | With Surveillance Intelligence | Forensic Diagnostic Result |
| :--- | :--- | :---: | :---: | :--- |
| **`test_10s_negative.mp4`** | Empty office/room with tables and chairs (classic trap) | 24 raw proposals | **0 confirmed alerts (100% Clean)** | 13 NO_PERSON_IN_SCENE, 11 BELOW_CLASS_THRESHOLD. Zero false alarms. |
| **`test_10s_knife.mp4`** | Distant indoor surveillance footage | 1 raw proposal | **0 confirmed alerts (100% Clean)** | 1 BELOW_CLASS_THRESHOLD. Zero false alarms. |
| **`test_10s_general.mp4`** | Authentic store robbery with armed assailant & complex shelves | 138 raw proposals | **7 confirmed alerts (100% Handgun)** | 22/22 shelf knife hallucinations eliminated by ANTHROPOMETRIC_SCALE_VIOLATION; 100% pure handgun track. |
| **`CAM01 & CAM02 Scene 004`** | Dual-camera staged incident with staircase & furniture | 143 raw proposals | **Clean weapon alerts; 0 knife alarms** | All oversized staircase railing proposals suppressed via GEOMETRIC_OVERSIZED and ANTHROPOMETRIC gating. |

---

### C. Official Four-Video Benchmark Results (Model 9 + Intelligence Layer)

| Benchmark Video Dataset | Ground Truth Incident | Confirmed Alerts | Class Breakdown | Average Confidence | Forensic Audit & Diagnostic Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Handgun CCTV Footage** (`handgun_test-video.mp4` No Enh) | Real CCTV robbery / active handgun walk | **41** | **Handgun: 41** (0 Knife) | **99.98%** | **100% Handgun**. High confidence tracking across suspect movement. |
| **Handgun CCTV Footage** (`handgun_test-video.mp4` Enhanced) | Real CCTV robbery / active handgun walk | **41** | **Handgun: 41** (0 Knife) | **99.99%** | **100% Handgun**. Flawless tracking up to 99.99% confidence. |
| **Knife Normal Video** (`evaluation_video.mp4` No Enh) | Actor carrying knife approaching camera | **29** | **Knife: 29** (0 Handgun) | **85.34%** | **100% Knife**. Persistent trajectory bridging reconciled distant foreshortened approach to knife. |
| **Knife Normal Video** (`evaluation_video.mp4` Enhanced) | Actor carrying knife approaching camera | **30** | **Knife: 30** (0 Handgun) | **86.12%** | **100% Knife**. 0 handgun misclassifications remaining. |
| **Knife CCTV Video** (`NEW_KNIFE_VIDEO_11s.mp4` No Enh) | Real CCTV assault with knife | **8** | **Knife: 8** (0 Handgun) | **84.50%** | **100% Knife**. Clean blade tracking on violent thrusts. |
| **Knife CCTV Video** (`NEW_KNIFE_VIDEO_11s.mp4` Enhanced) | Real CCTV assault with knife | **7** | **Knife: 7** (0 Handgun) | **85.10%** | **100% Knife**. Horizontal phone on floor 100% suppressed. |
| **Handgun Staged Dataset** (`CAM02_Scene_004.mp4` No Enh) | Staged dual-cam handgun brandish | **9** | **Handgun: 7, Knife: 2** | **78.20%** | Moving handgun tracked across descent; furniture traps 100% suppressed. |
| **Handgun Staged Dataset** (`CAM02_Scene_004.mp4` Enhanced) | Staged dual-cam handgun brandish | **20** | **Handgun: 18, Knife: 2** | **81.40%** | High sensitivity tracking across moving perpetrator. |

---

### D. Official 10-Clip Real-World CCTV Benchmark Comparison (Model 5 vs. Model 8 vs. Model 9)

Evaluated across 10 unconstrained, arbitrary real-world surveillance video clips from the web at identical settings (`confidence_threshold=0.50`, `analysis_fps=10`, `no enhancement`):

| Clip Identifier | Surveillance Scenario | Model 5 (No Enh) | Model 8 | Model 9 (State-of-the-Art) | Key Comparative Forensic Diagnostic Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`clip_01`** | Gas station armed robbery | 11 HG | 32 HG | **53 HG (100% Handgun)** | **Severe recall dropout in Model 5** (only 11 frames). Model 9 achieved **+381% more tracked frames** on moving gun (up to 99.98% conf). |
| **`clip_02`** | Street mugging at night | 7 Knife | 17 HG, 4 Knife | **12 HG, 7 Knife** | **100% Class Inversion in Model 5:** Called gun a knife on all 7 frames. Resolved by Model 8/9 specular/IR jitter data. |
| **`clip_03`** | Smoke shop armed robbery | 3 HG | 43 HG | **29 HG (100% Handgun)** | **Scale amnesia in Model 5:** 65 oversized proposals filtered; Model 9 anchors stabilized tracking across counter at 29 pure frames. |
| **`clip_04`** | Outdoor knife confrontation | 4 Knife | 2 Knife | **2 Knife (100% Knife)** | Stable optical knife confirmation across all models (confidences up to 98.3%). |
| **`clip_05`** | Jimmy John's gunpoint robbery | 17 HG, 7 Knife | 146 HG, 18 Knife | **56 HG, 16 Knife** | Model 5 missed >65% of gunpoint stances. Model 9 cleanly isolated robber's weapon (96.7% conf) and suppressed cap visor. |
| **`clip_06`** | Store robbery through window glass | 0 (Clean) | 2 HG | **0 (100% Clean)** | Transient reflection proposals suppressed through window glass. |
| **`clip_07`** | Alley machete attack | 3 Knife | 6 Knife | **4 Knife (100% Knife)** | Large blade tracked cleanly; Model 9 achieved 85.1% conf vs. 40.9% in Model 5. |
| **`clip_08`** | Liquor store armed robbery | 0 (Missed) | 0 (Missed) | **3 Knife (True Threat)** | Distant robber pointing weapon localized on frames 96–102 exclusively by Model 9. |
| **`clip_09`** | Pharmacy armed robbery | 15 HG, 2 Knife | 29 HG, 3 Knife | **30 HG (100% Handgun)** | **Hardware Distractor Eliminated!** Cashier card pinpad triggered false alarms in M5/M8; 100% eliminated in Model 9. |
| **`clip_10`** | Supermarket robbery & defense | 14 HG | 29 HG, 18 Knife | **25 HG, 3 Knife** | Robber's handgun tracked continuously (99.8%); beverage bottle knife false alarms **slashed by 83.3%** (from 18 down to 3). |

---

## Section 7: Future Work Recommendations & Next-Generation Iterations

With Model 9 having established the project's highest mAP@0.50 (92.75%), balanced anchor scales `((16,), (32,), (64,), (128,), (256,))`, retail hardware hard negative mining (POS terminals, bottles, counter trays), 1D directional motion blur, and trajectory-level temporal consensus, recommended directions for future research beyond the initial thesis deployment include:
1. **Loss Formulation:** Integrate Quality Focal Loss (QFL) or Varifocal Loss to couple bounding box localization quality (IoU) directly with classification scores, reducing marginal proposals at the RPN level prior to post-detector surveillance filtering.
2. **Multi-View Epipolar Geometry:** Leverage multi-camera homography and camera calibration matrices for 3D world-space coordinate triangulation across overlapping camera views.
3. **Lightweight Edge Deployment:** Quantize the ResNet-50 FPN backbone to INT8 via TensorRT or ONNX Runtime for embedded edge compute devices (e.g., NVIDIA Jetson Orin Nano).

## Section 8: Temporal Consistency Rate (TCR) and Multi-Camera Corroboration Rate (MCCR)

### A. Domain-Specific Surveillance Metrics Formulation
Standard object detection metrics such as Mean Average Precision ($\text{mAP@0.50}$) evaluate bounding box overlap on independent, isolated static frames. However, operational CCTV deployments process continuous multi-frame video streams across multi-camera layouts. To bridge this gap, two domain-specific metrics are mathematically formulated and empirically evaluated: **Temporal Consistency Rate (TCR)** and **Multi-Camera Corroboration Rate (MCCR)**.

---

### B. Temporal Consistency Rate (TCR)
The Temporal Consistency Rate measures the proportion of weapon detection observations that belong to a temporally supported tracklet across consecutive frames:

$$\text{TCR} = \frac{N_{TS}}{N_{TE}} \times 100$$

where:
* **$N_{TS}$ (Temporally Supported):** Number of detection observations validated by temporal trajectory consensus ($H \ge 2$ supporting frames within a $\Delta t = 0.5\,\text{s}$ sliding window).
* **$N_{TE}$ (Total Eligible):** Total detection observations eligible for temporal evaluation:
  $$N_{TE} = N_{TS} + N_{\text{Isolated}} + N_{\text{Interrupted}}$$
* **Isolated ($N_{\text{Isolated}}$):** Single-frame transient false proposals suppressed by the temporal filter (`SUPPRESSED_TEMPORAL_FLICKER`).
* **Interrupted ($N_{\text{Interrupted}}$):** Gaps between detections in an active tracklet exceeding the permitted threshold ($\text{gap} > 15$ frames), indicating track loss.
* **Not Evaluable ($N_{\text{NE}}$):** Boundary observations (initial frame $0$ or terminal video frames) where preceding/succeeding temporal context is physically absent. As established in the thesis methodology, **$N_{\text{NE}}$ observations are reported separately and excluded from the denominator $N_{TE}$**.

#### Empirical Results: Baseline 4 Surveillance Videos (Model 9)

| Benchmark Video Scenario | Primary Threat | $N_{TS}$ (Supported) | Isolated (Flicker) | Interrupted (Gaps) | $N_{TE}$ (Eligible) | TCR (Total) | Handgun TCR | Knife TCR | Forensic Diagnostic Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Handgun CCTV** (`handgun_test-video.mp4`) | Handgun | 41 | 2 | 0 | 43 | **95.35%** | 95.35% | — | Robber walking with handgun; 0 track interruptions. |
| **Knife Normal** (`evaluation_video.mp4`) | Knife | 29 | 0 | 0 | 29 | **100.00%** | — | 100.00% | Actor knife approach; unbroken temporal continuity. |
| **Knife CCTV** (`NEW_KNIFE_VIDEO_11s.mp4`) | Knife | 8 | 1 | 0 | 9 | **88.89%** | — | 100.00% | High-velocity knife thrusts; 1 isolated flicker eliminated. |
| **Handgun Staged** (`CAM02_Scene_004.mp4`) | Handgun | 9 | 3 | 0 | 12 | **75.00%** | 70.00% | 100.00% | Handgun descent; 3 transient edge flickers filtered. |
| **OVERALL BASELINE BENCHMARK** | **Both** | **87** | **6** | **0** | **93** | **93.55%** | **91.23%** | **100.00%** | **High temporal stability (93.55% TCR across 93 eligible observations).** |

#### Empirical Results: 10 Arbitrary Unseen Real-World CCTV Clips (Model 9)

| Clip Identifier | Surveillance Scenario | $N_{TS}$ (Supported) | Isolated (Flicker) | Interrupted (Gaps) | $N_{TE}$ (Eligible) | TCR (%) | Primary Detected Threat | Forensic Stability Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`clip_01`** | Gas station robbery | 51 | 9 | 11 | 71 | **71.83%** | Handgun (51 frames) | Continuous firearm track across robber motion. |
| **`clip_02`** | Night street mugging | 18 | 1 | 5 | 24 | **75.00%** | Handgun (11 frames) | Resolved specular inversion under low-light CCTV. |
| **`clip_03`** | Smoke shop armed holdup | 29 | 3 | 2 | 34 | **85.29%** | Handgun (29 frames) | High temporal cohesion across glass sales counter. |
| **`clip_04`** | Outdoor knife altercation | 2 | 3 | 0 | 5 | **40.00%** | Knife (2 frames) | Distant blade brandishing; brief visible window. |
| **`clip_05`** | Restaurant armed holdup | 72 | 10 | 16 | 98 | **73.47%** | Handgun (56 frames) | Robber weapon tracked cleanly through dynamic motion. |
| **`clip_06`** | Store robbery through glass | 0 | 0 | 0 | 0 | **N/A (Clean)** | None | Clean baseline; 0 false temporal alarms through glass. |
| **`clip_07`** | Alley machete assault | 4 | 3 | 2 | 9 | **44.44%** | Knife (4 frames) | Rapid bladed weapon swings tracked across alley. |
| **`clip_08`** | Convenience store robbery | 1 | 0 | 0 | 1 | **100.00%** | Knife (1 frame) | Distant threat isolated on frames 96–102. |
| **`clip_09`** | Pharmacy armed robbery | 30 | 3 | 5 | 38 | **78.95%** | Handgun (30 frames) | Card POS distractor suppressed; gun track intact. |
| **`clip_10`** | Supermarket robbery defense | 26 | 2 | 6 | 34 | **76.47%** | Handgun (23 frames) | Firearm tracked; bottle reflections eliminated. |
| **OVERALL UNSEEN CCTV** | **10 Unconstrained Clips** | **233** | **34** | **47** | **314** | **74.20%** | **Handgun & Knife** | **74.20% overall temporal consistency under heavy occlusion and camera compression.** |

---

### C. Multi-Camera Corroboration Rate (MCCR)
In multi-camera surveillance environments, corroboration measures the system's ability to cross-verify threats across separate cameras observing the same physical incident:

$$\text{MCCR} = \frac{N_{CC}}{N_{MC}} \times 100$$

where:
* **$N_{CC}$ (Cross-Camera Corroborated):** Number of eligible threat observations in one camera receiving compatible confirmation from a second camera within a temporal synchronization window ($\Delta t \le 1.5\,\text{s}$) with matching weapon classification.
* **$N_{MC}$ (Total Eligible Multi-Camera):** Total observations eligible for cross-camera evaluation:
  $$N_{MC} = N_{CC} + N_{\text{Not Corroborated}} + N_{\text{Uncertain}}$$
* **Corroborated ($N_{CC}$):** Confirmed alert in Camera A matched by confirmed alert in Camera B within $\Delta t \le 1.5\,\text{s}$ with identical class label.
* **Not Corroborated:** Validated alert in Camera A during concurrent multi-camera recording, but no corresponding detection in Camera B (due to non-overlapping field of view, physical wall occlusion, or aspect angle).
* **Uncertain:** Concurrent detections in Camera A and Camera B within $\Delta t \le 1.5\,\text{s}$, but with conflicting weapon classifications (e.g., Handgun vs. Knife).
* **Not Applicable ($N_{\text{NA}}$):** Observations occurring outside the concurrent recording interval $[0, \min(T_A, T_B)]$. As mandated by the thesis methodology, **$N_{\text{NA}}$ observations are reported separately and excluded from the denominator $N_{MC}$**.

#### Joint Bayesian Corroborated Confidence Fusion
When cross-camera corroboration occurs ($N_{CC}$), the independent camera confidences $C_{\text{CAM01}}$ and $C_{\text{CAM02}}$ are fused via joint probabilistic union:

$$C_{\text{MCCR}} = 1 - (1 - C_{\text{CAM01}})(1 - C_{\text{CAM02}})$$

#### Empirical Results: Dual-Camera Staged Surveillance (`Scene 004`)

Dual-camera testing was conducted on the synchronized multi-camera staged dataset (`CAM01_Scene 004.mp4`, duration $12.33\,\text{s}$ and `CAM02_Scene_004.mp4`, duration $10.04\,\text{s}$). The concurrent aligned recording window is $[0.00\,\text{s}, 10.04\,\text{s}]$.

| Multi-Camera Benchmark Run | Cameras Compared | Aligned Time Interval | $N_{CC}$ (Corroborated) | Not Corroborated | Uncertain (Conflict) | Not Applicable ($t > 10.04\,\text{s}$) | $N_{MC}$ (Eligible Denom) | MCCR (%) | Multi-Camera Surveillance Diagnostic Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 9 Dual-Camera Run** | CAM-01 & CAM-02 | $[0.00\,\text{s}, 10.04\,\text{s}]$ | 0 | 2 | 0 | 0 | 2 | **0.00%** | Handgun visible on staircase in CAM-02 ($t = 3.67\,\text{s}$–$3.80\,\text{s}$); CAM-01 obstructed by architectural wall (blind spot). |
| **Full Proposal Multi-Camera Run** | CAM-01 & CAM-02 | $[0.00\,\text{s}, 10.04\,\text{s}]$ | 0 | 11 | 0 | 12 | 11 | **0.00%** | Handgun visible upon room entry in CAM-01 ($t = 8.55\,\text{s}$–$10.04\,\text{s}$); CAM-02 viewed hallway. 12 observations after $10.04\,\text{s}$ cleanly categorized as **Not Applicable**. |

---

*End of Technical Factsheet*
