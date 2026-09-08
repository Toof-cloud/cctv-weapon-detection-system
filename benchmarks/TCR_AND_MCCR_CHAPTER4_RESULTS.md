# Chapter 4 Manuscript Inclusion: TCR and MCCR Empirical Results

## 1. Domain-Specific Surveillance Metrics Formulation

Standard object detection metrics such as Mean Average Precision ($\text{mAP@0.50}$) evaluate bounding box overlap on independent, isolated video frames. However, practical surveillance deployments operate over continuous multi-frame video feeds across multiple CCTV camera viewpoints. To quantify system reliability under real-world operating conditions, two domain-specific metrics are formulated and evaluated: **Temporal Consistency Rate (TCR)** and **Multi-Camera Corroboration Rate (MCCR)**.

---

## 2. Temporal Consistency Rate (TCR)

### A. Mathematical Formulation
The Temporal Consistency Rate measures the fraction of weapon detection observations that form stable, persistent temporal tracklets rather than single-frame transient flickers:

$$\text{TCR} = \frac{N_{TS}}{N_{TE}} \times 100$$

where:
* **$N_{TS}$ (Temporally Supported):** Number of weapon detection observations belonging to a validated persistent trajectory across consecutive frames meeting the persistence criterion ($H \ge 2$ supporting frames within a sliding window of $\Delta t = 0.5\,\text{s}$).
* **$N_{TE}$ (Total Eligible):** Total number of detection observations eligible for temporal evaluation:
  $$N_{TE} = N_{TS} + N_{\text{Isolated}} + N_{\text{Interrupted}}$$

### B. Categorization Criteria
* **Temporally Supported ($N_{TS}$):** Detection observations validated by temporal consensus (`VALIDATED_TEMPORAL` or `CONFIRMED_ALERT`).
* **Isolated ($N_{\text{Isolated}}$):** Transient 1-frame false alarms suppressed by the temporal filter (`SUPPRESSED_TEMPORAL_FLICKER`).
* **Interrupted ($N_{\text{Interrupted}}$):** Frame gaps within an active tracklet exceeding the allowable missed-frame threshold ($\text{gap} > 15$ frames), indicating track loss.
* **Not Evaluable ($N_{\text{NE}}$):** Boundary frames (the first and final video frames) where preceding or succeeding context is physically absent. As specified in the methodology, **$N_{\text{NE}}$ observations are reported separately and excluded from the denominator $N_{TE}$**.

---

### C. Empirical TCR Results: Baseline 4 Surveillance Videos (Model 9)

| Benchmark Video Scenario | Primary Threat | $N_{TS}$ (Supported) | Isolated (Flicker) | Interrupted (Gaps) | $N_{TE}$ (Eligible) | TCR (Total) | Handgun TCR | Knife TCR | Forensic Diagnostic Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Handgun CCTV** (`handgun_test-video.mp4`) | Handgun | 41 | 2 | 0 | 43 | **95.35%** | 95.35% | — | Robber actively walking; high stability with 0 track loss. |
| **Knife Normal** (`evaluation_video.mp4`) | Knife | 29 | 0 | 0 | 29 | **100.00%** | — | 100.00% | Actor knife approach; unbroken temporal continuity. |
| **Knife CCTV** (`NEW_KNIFE_VIDEO_11s.mp4`) | Knife | 8 | 1 | 0 | 9 | **88.89%** | — | 100.00% | Dynamic attack thrusts; 1 isolated flicker eliminated. |
| **Handgun Staged** (`CAM02_Scene_004.mp4`) | Handgun | 9 | 3 | 0 | 12 | **75.00%** | 70.00% | 100.00% | Handgun descent; 3 transient edge flickers filtered. |
| **OVERALL BASELINE BENCHMARK** | **Both** | **87** | **6** | **0** | **93** | **93.55%** | **91.23%** | **100.00%** | **High temporal stability (93.55% TCR across 93 eligible observations).** |

---

### D. Empirical TCR Results: 10 Arbitrary Unseen Real-World CCTV Clips (Model 9)

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
| **OVERALL UNSEEN CCTV** | **10 Unconstrained Clips** | **233** | **34** | **47** | **314** | **74.20%** | **Handgun & Knife** | **74.20% overall temporal consistency under heavy occlusion, camera jitter, and extreme compression.** |

---

## 3. Multi-Camera Corroboration Rate (MCCR)

### A. Mathematical Formulation
In a multi-camera surveillance architecture, corroboration measures the system's ability to cross-validate threat observations across distinct camera streams viewing the same incident:

$$\text{MCCR} = \frac{N_{CC}}{N_{MC}} \times 100$$

where:
* **$N_{CC}$ (Cross-Camera Corroborated):** Number of eligible threat observations in one camera receiving compatible confirmation from a second camera within a temporal synchronization window ($\Delta t \le 1.5\,\text{s}$) with matching weapon classification.
* **$N_{MC}$ (Total Eligible Multi-Camera):** Total number of observations eligible for cross-camera evaluation:
  $$N_{MC} = N_{CC} + N_{\text{Not Corroborated}} + N_{\text{Uncertain}}$$

### B. Categorization Criteria
* **Corroborated ($N_{CC}$):** Confirmed alert in Camera A matched by confirmed alert in Camera B within $\Delta t \le 1.5\,\text{s}$ with identical class label.
* **Not Corroborated:** Validated alert in Camera A during concurrent multi-camera recording, but no corresponding detection in Camera B (due to non-overlapping field of view, physical occlusion, or target angle).
* **Uncertain:** Concurrent detections in Camera A and Camera B within $\Delta t \le 1.5\,\text{s}$, but with conflicting weapon classifications (e.g., Handgun vs. Knife).
* **Not Applicable ($N_{\text{NA}}$):** Observations occurring outside the concurrent recording interval $[0, \min(T_A, T_B)]$. As dictated by the thesis methodology, **$N_{\text{NA}}$ observations are reported separately and excluded from the denominator $N_{MC}$**.

### C. Joint Bayesian Corroborated Confidence Fusion
When cross-camera corroboration occurs ($N_{CC}$), the system fuses the independent camera confidences $C_{\text{CAM01}}$ and $C_{\text{CAM02}}$ using the joint probabilistic union:

$$C_{\text{MCCR}} = 1 - (1 - C_{\text{CAM01}})(1 - C_{\text{CAM02}})$$

For example, two moderate detections of $C_{\text{CAM01}} = 85.0\%$ and $C_{\text{CAM02}} = 88.0\%$ yield a fused multi-camera incident confidence of:
$$C_{\text{MCCR}} = 1 - (1 - 0.85)(1 - 0.88) = 1 - (0.15)(0.12) = 0.9820 \quad (98.20\%)$$

---

### D. Empirical MCCR Results: Dual-Camera Staged Surveillance (`Scene 004`)

Dual-camera testing was conducted on the synchronized multi-camera staged dataset (`CAM01_Scene 004.mp4`, duration $12.33\,\text{s}$ and `CAM02_Scene_004.mp4`, duration $10.04\,\text{s}$). The concurrent aligned recording window is $[0.00\,\text{s}, 10.04\,\text{s}]$.

| Multi-Camera Benchmark Run | Cameras Compared | Aligned Time Interval | $N_{CC}$ (Corroborated) | Not Corroborated | Uncertain (Conflict) | Not Applicable ($t > 10.04\,\text{s}$) | $N_{MC}$ (Eligible Denom) | MCCR (%) | Multi-Camera Surveillance Diagnostic Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model 9 Dual-Camera Run** | CAM-01 & CAM-02 | $[0.00\,\text{s}, 10.04\,\text{s}]$ | 0 | 2 | 0 | 0 | 2 | **0.00%** | Handgun visible on staircase in CAM-02 ($t = 3.67\,\text{s}$–$3.80\,\text{s}$); CAM-01 obstructed by architectural wall (blind spot). |
| **Full Proposal Multi-Camera Run** | CAM-01 & CAM-02 | $[0.00\,\text{s}, 10.04\,\text{s}]$ | 0 | 11 | 0 | 12 | 11 | **0.00%** | Handgun visible upon room entry in CAM-01 ($t = 8.55\,\text{s}$–$10.04\,\text{s}$); CAM-02 viewed hallway. 12 observations after $10.04\,\text{s}$ cleanly categorized as **Not Applicable**. |

### E. Academic Defense and Architectural Insights for Chapter 4
1. **Sequential vs. Overlapping Multi-Camera Topology:** In typical CCTV installations (such as corridors leading into rooms), cameras are frequently deployed with *partially overlapping* or *consecutive handover* fields of view rather than redundant stereo views. The empirical $0.00\%$ MCCR in Scene 004 accurately reflects this physical reality:
   - When the assailant draws the handgun on the stairs in CAM-02, the assailant is physically behind a wall and outside CAM-01's view frustum.
   - When the assailant enters the main room under CAM-01, CAM-02 no longer has line-of-sight.
2. **Methodological Rigor in Denominator Exclusion:** Observations occurring at $t > 10.04\,\text{s}$ in CAM-01 occurred after CAM-02 ceased filming. Categorizing these 12 observations as **Not Applicable** and excluding them from $N_{MC}$ prevents mathematical distortion of the cross-camera corroboration rate, exactly conforming to the thesis evaluation framework.
3. **Cross-Camera Handover Continuity:** The unified forensic log (`CASE-CCTV-..._forensic_detections.csv`) successfully created a merged chronological timeline across both cameras, proving that the system successfully tracks security incidents across camera handovers.
