# Thesis 2 — Pre-Fix TODO

Things to clear out **before** starting the panel fixes (false negatives + look-alike negatives).
See also: [PANEL_RECOMMENDATIONS.md](PANEL_RECOMMENDATIONS.md)

## 1. Repo housekeeping
- [ ] Decide on the 10 uncommitted Mac edits (Python 3.9 compatibility: `from __future__ import annotations`, `zip(strict=)` removal, `hashlib.file_digest` fallback, `enterContext` shim) — commit to a branch or discard.
- [ ] Decide on untracked `dataset_analysis/download_reviewer_b.py` (commit or delete).
- [ ] Commit `docs/PANEL_RECOMMENDATIONS.md` and this TODO.
- [ ] On the PC: `git pull` so both machines are on `c68dd89`+.
- [ ] Clean stray folders in repo root: `-p/`, `lines/`, `more/` (accidental `mkdir` artifacts).
- [ ] Fix naming: V11 is now "Model 9" — the old Model 9 still exists in the manuscript table. Pick final names (e.g., Model 9 = old baseline, Model 10 = V11) and apply consistently in code, docs, and manuscript.

## 2. Adviser meeting (get approval before collecting data)
- [ ] Present panel concerns with corrected wording (false **negatives**: 63/238 missed, 59 are knives).
- [ ] Ask: is a staged airsoft/toy/prop weapon dataset on our CCTV acceptable?
- [ ] Ask: staged data train-only, or also a separate staged test set?
- [ ] Ask: staged hard negatives (pen, eggplant, phone, etc.) acceptable if online images are insufficient?
- [ ] Ask: consent forms / campus permission for replica weapons / filming location.
- [ ] Ask: do we need to re-evaluate all previous models on one fixed test set (see §3)?
- [ ] Agree on target metrics (proposal: knife recall ≥ 80%, look-alike FP rate ≤ 5%).

## 3. Test split audit (on PC — data folders are gitignored)
- [ ] Locate every `training/model_*/<name>_data/test_ids.txt` and `ninth_model_data`.
- [ ] Build overlap matrix of test sets across Models 1–9 & V11.
- [ ] Check train/test leakage (same image or same video clip in both splits) per model.
- [ ] Confirm Models 5–7 share the same 353 weapon boxes (only negatives differ).
- [ ] Confirm Model 8 vs Model 9 test difference (265 vs 279 images, both 238 boxes).
- [ ] Freeze **one benchmark test set** (recommend Model 9 / V11's 279-image set) — never train on it.
- [ ] Re-evaluate saved checkpoints (Model 5 → V11) on the frozen benchmark for a fair Chapter 4 table.

## 4. Baseline diagnosis of the false negatives
- [ ] Export the 63 V11 missed boxes (59 knife, 4 handgun) as image crops.
- [ ] Categorize misses: distance/size, blur, lighting, occlusion, blade orientation, low confidence vs. no proposal.
- [ ] Check if misses are threshold-related: sweep knife score threshold, plot precision-recall curve.
- [ ] Compare against Model 9 (old) on the same images — which knives did V11 lose and why (anchors, clamp, negatives)?
- [ ] Record findings in `docs/CHAPTER4_EXPERIMENT_LOG.md`.

## 5. Environment readiness
- [ ] Confirm PC GPU training env still works (RTX 5060 Ti, torch/CUDA, thread caps).
- [ ] Confirm V11 checkpoint `best_candidate_model_v11.pth` and Model 9 `.pth` are backed up.
- [ ] Back up all `*_data/` dataset folders and hard-negative folders (external drive / cloud).
- [ ] Decide Mac role: docs/scripts only, or also local inference (CPU/MPS) for quick checks.

## 6. Data collection planning (after adviser approval)
- [ ] Buy/borrow props: realistic airsoft/replica handgun (cover orange tip), metal-look prop knife(s).
- [ ] Look-alike objects list: pen, eggplant, phone, remote, flashlight, spatula, comb, stapler, wallet, umbrella, hair dryer, banana, scissors, empty hands.
- [ ] Shot list: distances (near/mid/far), angles (ceiling/eye-level), lighting (day/night/IR), motions (walk, draw, swing, point), occlusion.
- [ ] Mirror the same poses for weapons and look-alikes.
- [ ] Choose annotation tool (CVAT / Label Studio / Roboflow) and box guidelines.
- [ ] Plan split **by recording session**, not by frame.
- [ ] Search online sources first for look-alike negatives (Open Images classes, Kaggle); log sources & licenses.

## 7. Documentation / manuscript
- [ ] Update manuscript Chapter 3 methodology for staged data + hard negatives.
- [ ] Add a note on test-split differences across Models 1–9 (or replace with re-evaluated table).
- [ ] Add the new look-alike FP rate metric definition.
- [ ] Refresh `docs/AI_HANDOFF_CONTEXT.md` (last updated 2026-08-22; outdated paths and status).

---
**Ready to start fixes when:** §1 done, adviser approval in §2, benchmark frozen in §3, FN diagnosis in §4 complete.
