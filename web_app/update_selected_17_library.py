"""
Update the FORENSIKADA Web App Prepared Evaluation Library (`web_app/library_data/`)
from a clean, editable selection list of the 17 chosen videos across the 5 categories:
  1. 4 CCTV Handgun videos
  2. 4 CCTV Knife videos
  3. 3 Staged Multi-Camera weapon pairs (CAM-01 & CAM-02)
  4. 3 Non-Handgun Staged Dataset videos (plus the synchronized Smartphone Multi-Camera Non-Weapon pair)
  5. 3 Non-Handgun Convenience Store / Real CCTV videos

How to change or swap any video later:
  1. Open this file (`web_app/update_selected_17_library.py`)
  2. Edit `SELECTED_SINGLE_VIDEOS` or `SELECTED_MULTICAM_SCENES` using any `candidate_name`
     from `outputs/00_GOOGLE_DRIVE_MODEL12_ANNOTATED_VIDEOS_FOR_SELECTION/ALL_CANDIDATES_MODEL12_LEADERBOARD.csv`
  3. Run:
       .\.venv\Scripts\python.exe web_app/update_selected_17_library.py
  4. Push to GitHub (`git add web_app/library_data && git commit -m "Update web videos" && git push`)
     — Streamlit Community Cloud updates your live 24/7 URL automatically!
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.annotate_all_candidates_for_drive import build_candidate_manifest

DRIVE_ROOT = ROOT / "outputs" / "00_GOOGLE_DRIVE_MODEL12_ANNOTATED_VIDEOS_FOR_SELECTION"
WEB_DATA_DIR = ROOT / "web_app" / "library_data"
WEB_VIDEOS_DIR = WEB_DATA_DIR / "videos"
WEB_TELEM_DIR = WEB_DATA_DIR / "telemetry"


def find_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    winget_dir = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if winget_dir.exists():
        for p in winget_dir.rglob("ffmpeg.exe"):
            return str(p)
    return "ffmpeg"


FFMPEG_BIN = find_ffmpeg()

# ==============================================================================
# EDIT THIS LIST ANYTIME IF YOUR GROUPMATES WANT TO SWAP ANY OF THE 17 VIDEOS
# ==============================================================================
SELECTED_SINGLE_VIDEOS = [
    # --- 1. 4 CCTV HANDGUN VIDEOS (plus Original counterparts for BasicVSR++ toggle where available) ---
    {
        "video_id": "HG01_Enhanced",
        "pair_id": "HG01",
        "candidate_name": "HG01_230249_StoreRobbery_Enhanced",
        "title": "CCTV Handgun 1 — Store Counter Robbery (230249) [Enhanced]",
        "category": "Handgun",
        "version": "Enhanced",
    },
    {
        "video_id": "HG01_Original",
        "pair_id": "HG01",
        "candidate_name": "HG01_230249_StoreRobbery_Original",
        "title": "CCTV Handgun 1 — Store Counter Robbery (230249) [Original]",
        "category": "Handgun",
        "version": "Original",
    },
    {
        "video_id": "HG02_Enhanced",
        "pair_id": "HG02",
        "candidate_name": "HG03_HallwayHandgunDraw_Enhanced",
        "title": "CCTV Handgun 2 — Indoor Hallway Handgun Draw [Enhanced]",
        "category": "Handgun",
        "version": "Enhanced",
    },
    {
        "video_id": "HG02_Original",
        "pair_id": "HG02",
        "candidate_name": "HG03_HallwayHandgunDraw_Original",
        "title": "CCTV Handgun 2 — Indoor Hallway Handgun Draw [Original]",
        "category": "Handgun",
        "version": "Original",
    },
    {
        "video_id": "HG03_Enhanced",
        "pair_id": "HG03",
        "candidate_name": "CCTV_HG_SR_2026-09-10_213824-HANDGUN_41c923d345d9_",
        "title": "CCTV Handgun 3 — Retail Surveillance Armed Suspect (213824) [Enhanced]",
        "category": "Handgun",
        "version": "Enhanced",
    },
    {
        "video_id": "HG04_Enhanced",
        "pair_id": "HG04",
        "candidate_name": "HG02_213340_StreetHandgun_Enhanced",
        "title": "CCTV Handgun 4 — Elevated Night Street Surveillance (213340) [Enhanced]",
        "category": "Handgun",
        "version": "Enhanced",
    },
    {
        "video_id": "HG04_Original",
        "pair_id": "HG04",
        "candidate_name": "HG02_213340_StreetHandgun_Original",
        "title": "CCTV Handgun 4 — Elevated Night Street Surveillance (213340) [Original]",
        "category": "Handgun",
        "version": "Original",
    },

    # --- 2. 4 CCTV KNIFE VIDEOS ---
    {
        "video_id": "KN01_Enhanced",
        "pair_id": "KN01",
        "candidate_name": "KN02_220101_StreetKnife_Enhanced",
        "title": "CCTV Knife 1 — Corridor & Street Blade Assault (220101) [Enhanced]",
        "category": "Knife",
        "version": "Enhanced",
    },
    {
        "video_id": "KN01_Original",
        "pair_id": "KN01",
        "candidate_name": "KN02_220101_StreetKnife_Original",
        "title": "CCTV Knife 1 — Corridor & Street Blade Assault (220101) [Original]",
        "category": "Knife",
        "version": "Original",
    },
    {
        "video_id": "KN02_Enhanced",
        "pair_id": "KN02",
        "candidate_name": "CCTV_KN_SR_2026-09-10_220510-KNIFE_de569e413ac3_en",
        "title": "CCTV Knife 2 — Close-Quarters Blade Threat (220510) [Enhanced]",
        "category": "Knife",
        "version": "Enhanced",
    },
    {
        "video_id": "KN03_Enhanced",
        "pair_id": "KN03",
        "candidate_name": "KN01_230047_FastKnifeSwing_Enhanced",
        "title": "CCTV Knife 3 — Fast Indoor Blade Swing (230047) [Enhanced]",
        "category": "Knife",
        "version": "Enhanced",
    },
    {
        "video_id": "KN03_Original",
        "pair_id": "KN03",
        "candidate_name": "KN01_230047_FastKnifeSwing_Original",
        "title": "CCTV Knife 3 — Fast Indoor Blade Swing (230047) [Original]",
        "category": "Knife",
        "version": "Original",
    },
    {
        "video_id": "KN04_Enhanced",
        "pair_id": "KN04",
        "candidate_name": "CCTV_KN_SR_2026-09-10_215822-KNIFE_b0e9b0697664_en",
        "title": "CCTV Knife 4 — Overhead Surveillance Knife Detection (215822) [Enhanced]",
        "category": "Knife",
        "version": "Enhanced",
    },

    # --- 3. 3 NON-HANDGUN STAGED DATASET VIDEOS (0 FP) ---
    {
        "video_id": "NW_STAGED_01",
        "pair_id": "NW_STAGED_01",
        "candidate_name": "NOV20_Pair05_Smartphone_CAM01",
        "title": "Non-Handgun Staged 1 — Handheld Smartphone Reflections (CAM-01)",
        "category": "No-Weapon",
        "version": "Original",
    },
    {
        "video_id": "NW_STAGED_02",
        "pair_id": "NW_STAGED_02",
        "candidate_name": "NOV20_Pair05_Smartphone_CAM02",
        "title": "Non-Handgun Staged 2 — Handheld Smartphone Reflections (CAM-02)",
        "category": "No-Weapon",
        "version": "Original",
    },
    {
        "video_id": "NW_STAGED_03",
        "pair_id": "NW_STAGED_03",
        "candidate_name": "NOV20_Pair06_Wallet_CAM02",
        "title": "Non-Handgun Staged 3 — Dark Folded Wallet Picked Up From Table",
        "category": "No-Weapon",
        "version": "Original",
    },

    # --- 4. 3 NON-HANDGUN CONVENIENCE STORE VIDEOS (0 FP) ---
    {
        "video_id": "NW_STORE_01",
        "pair_id": "NW_STORE_01",
        "candidate_name": "NW01_Store_Customer_Interaction_Sample1",
        "title": "Non-Handgun Convenience Store 1 — Customer Interaction & Handheld Items",
        "category": "No-Weapon",
        "version": "Original",
    },
    {
        "video_id": "NW_STORE_02",
        "pair_id": "NW_STORE_02",
        "candidate_name": "NW02_Convenience_Store_Counter_Sample2",
        "title": "Non-Handgun Convenience Store 2 — Counter Drinks, Cups & Phones",
        "category": "No-Weapon",
        "version": "Original",
    },
    {
        "video_id": "NW_STORE_03",
        "pair_id": "NW_STORE_03",
        "candidate_name": "CCTV_NW_StoreCounter_SR_224743",
        "title": "Non-Handgun Convenience Store 3 — Retail Checkout Counter (224743)",
        "category": "No-Weapon",
        "version": "Original",
    },
]

# --- 5. 3 STAGED MULTI-CAMERA WEAPON PAIRS (+ 1 MULTI-CAMERA NON-WEAPON PAIR) ---
SELECTED_MULTICAM_SCENES = [
    {
        "scene_key": "MC_PAIR01_Enhanced",
        "scene_id": "MC_PAIR01",
        "title": "Multi-Camera Pair 1 — Staged Weapon Transition Scene 007 (CAM 01 & CAM 02) [Enhanced]",
        "category": "Multi-Camera Weapon",
        "version": "Enhanced",
        "cam1_candidate": "STAGED_SCENE007_CAM1_Enhanced",
        "cam2_candidate": "STAGED_SCENE007_CAM2_Enhanced",
    },
    {
        "scene_key": "MC_PAIR02_Enhanced",
        "scene_id": "MC_PAIR02",
        "title": "Multi-Camera Pair 2 — Staged Suspect Transition Scene 012 (CAM 01 & CAM 02) [Enhanced]",
        "category": "Multi-Camera Weapon",
        "version": "Enhanced",
        "cam1_candidate": "MC03_SCENE012_Knife_CAM01_Enhanced",
        "cam2_candidate": "MC03_SCENE012_Knife_CAM02_Enhanced",
    },
    {
        "scene_key": "MC_PAIR02_Original",
        "scene_id": "MC_PAIR02",
        "title": "Multi-Camera Pair 2 — Staged Suspect Transition Scene 012 (CAM 01 & CAM 02) [Original]",
        "category": "Multi-Camera Weapon",
        "version": "Original",
        "cam1_candidate": "MC03_SCENE012_Knife_CAM01_Original",
        "cam2_candidate": "MC03_SCENE012_Knife_CAM02_Original",
    },
    {
        "scene_key": "MC_PAIR03_Original",
        "scene_id": "MC_PAIR03",
        "title": "Multi-Camera Pair 3 — 1080p Dual-Tapo Kitchen Knife Corroboration (CAM 01 & CAM 02)",
        "category": "Multi-Camera Knife",
        "version": "Original",
        "cam1_candidate": "NOV20_Pair08_KitchenKnife_CAM01",
        "cam2_candidate": "NOV20_Pair08_KitchenKnife_CAM02",
    },
    {
        "scene_key": "MC_NONWEAPON_Original",
        "scene_id": "MC_NONWEAPON",
        "title": "Multi-Camera Non-Weapon Pair — Staged Smartphone Distractor (CAM 01 & CAM 02) [0 FP]",
        "category": "Multi-Camera No-Weapon",
        "version": "Original",
        "cam1_candidate": "NOV20_Pair05_Smartphone_CAM01",
        "cam2_candidate": "NOV20_Pair05_Smartphone_CAM02",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "width": w,
        "height": h,
        "fps": round(fps, 3),
        "total_frames": n,
        "duration_seconds": round(n / max(1.0, fps), 2),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def transcode_to_web_h264(src: Path, dst: Path, max_width: int = 960, crf: int = 26) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        f"scale='min({max_width},iw)':-2",
        "-crf",
        str(crf),
        "-preset",
        "fast",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {res.stderr.decode('utf-8', errors='ignore')[:400]}")


def parse_detections_csv(csv_path: Path, orig_w: int, orig_h: int) -> list[dict]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    records = []
    for _, r in df.iterrows():
        try:
            bx = ast.literal_eval(str(r["bounding_box"]))
            bx = [int(v) for v in bx]
        except Exception:
            bx = [0, 0, 0, 0]
        rej = str(r["rejection_reason"]).strip() if pd.notna(r["rejection_reason"]) else ""
        if rej.lower() == "nan":
            rej = ""
        records.append({
            "frame_number": int(r["frame_number"]),
            "timestamp_seconds": round(float(r["timestamp_seconds"]), 3),
            "timestamp_formatted": str(r["timestamp_formatted"]),
            "camera_id": str(r["camera_id"]),
            "object_label": str(r["object_label"]).lower().strip(),
            "bounding_box": bx,
            "norm_box": [
                round(bx[0] / max(1, orig_w), 5),
                round(bx[1] / max(1, orig_h), 5),
                round(bx[2] / max(1, orig_w), 5),
                round(bx[3] / max(1, orig_h), 5),
            ],
            "confidence_score": round(float(r["confidence_score"]), 4),
            "validation_status": str(r["validation_status"]),
            "rejection_reason": rej,
        })
    return records


def build_entry_from_drive_candidate(
    vid_id: str,
    pair_id: str,
    title: str,
    category: str,
    version: str,
    camera_id: str,
    cand_info: dict,
) -> dict:
    src_path: Path = cand_info["src_path"]
    cat_folder: str = cand_info["category"]
    out_name: str = cand_info["out_name"]

    annot_mp4_src = DRIVE_ROOT / cat_folder / f"{out_name}_M12_annotated.mp4"
    csv_src = DRIVE_ROOT / cat_folder / f"{out_name}_M12_detections.csv"
    if not annot_mp4_src.exists() or not csv_src.exists():
        raise FileNotFoundError(f"Missing pre-annotated Model 12 files for {out_name} in {DRIVE_ROOT / cat_folder}")

    orig_web_mp4 = WEB_VIDEOS_DIR / f"{vid_id}_original.mp4"
    annot_web_mp4 = WEB_VIDEOS_DIR / f"{vid_id}_annotated.mp4"
    telem_path = WEB_TELEM_DIR / f"{vid_id}.json"

    print(f"  -> Packaging {vid_id} <= {out_name} ({src_path.name})", flush=True)
    transcode_to_web_h264(src_path, orig_web_mp4, max_width=960, crf=26)
    transcode_to_web_h264(annot_mp4_src, annot_web_mp4, max_width=960, crf=26)

    orig_meta = get_video_meta(src_path)
    web_meta = get_video_meta(orig_web_mp4)
    file_hash = sha256_file(src_path)
    recs = parse_detections_csv(csv_src, orig_meta["width"], orig_meta["height"])

    entry = {
        "video_id": vid_id,
        "pair_id": pair_id,
        "title": title,
        "category": category,
        "version": version,
        "camera_id": camera_id,
        "source_filename": src_path.name,
        "sha256": file_hash,
        "orig_meta": orig_meta,
        "web_meta": web_meta,
        "original_video_rel": f"videos/{orig_web_mp4.name}",
        "annotated_video_rel": f"videos/{annot_web_mp4.name}",
        "telemetry_rel": f"telemetry/{telem_path.name}",
    }

    telem_payload = {
        "entry": entry,
        "model_name": "Model 12 (Final Production SOTA)",
        "model_checkpoint": "best_weapon_detector_twelfth_model.pth",
        "records_50pct": recs,
        "records_20pct": recs,
    }
    WEB_TELEM_DIR.mkdir(parents=True, exist_ok=True)
    telem_path.write_text(json.dumps(telem_payload, indent=2), encoding="utf-8")
    return entry


def main() -> None:
    manifest = {c["out_name"]: c for c in build_candidate_manifest()}

    if WEB_VIDEOS_DIR.exists():
        shutil.rmtree(WEB_VIDEOS_DIR)
    if WEB_TELEM_DIR.exists():
        shutil.rmtree(WEB_TELEM_DIR)
    WEB_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_TELEM_DIR.mkdir(parents=True, exist_ok=True)

    single_entries = []
    for item in SELECTED_SINGLE_VIDEOS:
        cname = item["candidate_name"]
        if cname not in manifest:
            raise KeyError(f"Unknown candidate_name: {cname}")
        entry = build_entry_from_drive_candidate(
            vid_id=item["video_id"],
            pair_id=item["pair_id"],
            title=item["title"],
            category=item["category"],
            version=item["version"],
            camera_id=item["pair_id"],
            cand_info=manifest[cname],
        )
        single_entries.append(entry)

    multicam_scenes = []
    for sc in SELECTED_MULTICAM_SCENES:
        c1_name = sc["cam1_candidate"]
        c2_name = sc["cam2_candidate"]
        if c1_name not in manifest or c2_name not in manifest:
            raise KeyError(f"Unknown multi-cam candidate: {c1_name} or {c2_name}")
        sk = sc["scene_key"]
        sid = sc["scene_id"]
        ver = sc["version"]
        e1 = build_entry_from_drive_candidate(
            vid_id=f"{sk}_CAM01",
            pair_id=sid,
            title=f"{sc['title']} — CAM-01",
            category=sc["category"],
            version=ver,
            camera_id="CAM-01",
            cand_info=manifest[c1_name],
        )
        e2 = build_entry_from_drive_candidate(
            vid_id=f"{sk}_CAM02",
            pair_id=sid,
            title=f"{sc['title']} — CAM-02",
            category=sc["category"],
            version=ver,
            camera_id="CAM-02",
            cand_info=manifest[c2_name],
        )
        multicam_scenes.append({
            "scene_key": sk,
            "scene_id": sid,
            "title": sc["title"],
            "category": sc["category"],
            "version": ver,
            "cam1": e1,
            "cam2": e2,
        })

    index_payload = {
        "system_name": "FORENSIKADA",
        "default_model": "Model 12 (Final Production SOTA)",
        "single_videos": single_entries,
        "multicam_scenes": multicam_scenes,
    }
    (WEB_DATA_DIR / "library_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in WEB_VIDEOS_DIR.glob("*.mp4")) / (1024 * 1024)
    print(f"\n[DONE] Indexed {len(single_entries)} single videos + {len(multicam_scenes)} multi-camera scenes ({total_mb:.2f} MB) -> {WEB_DATA_DIR}", flush=True)


if __name__ == "__main__":
    main()
