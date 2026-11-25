"""
Build the Final FORENSIKADA Web App Prepared Evaluation Library (`web_app/library_data/`):
  - 3 Single-Cam CCTV Handgun Scenes (HG01 [Enhanced + Original], HG02 [214325], HG03 [InStore Handgun])
  - 3 Single-Cam CCTV Knife Scenes (KN01 [220101 Original + Enhanced], KN02 [220510], KN03 [230047 Enhanced + Original])
  - 3 Single-Cam Non-Weapon Scenes (NW01 Convenience Store 1, NW02 Convenience Store 2, NW03 Staged Handheld Distractor — ALL 0 FP / 0 ghost boxes)
  - 3 Multi-Camera Weapon Pairs (Scene 007 Dual Knives, Scene 012 Handgun, Scene 001 Kitchen Knife — ALL corroborated on CAM-01 & CAM-02)
  - 1 Multi-Camera Non-Weapon Pair (Nov 20 Pair 05 Smartphone CAM-01 & CAM-02 — 0 FP / 0 ghost boxes)
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
# FINAL APPROVED SELECTION:
#   3 Single-Cam Handgun + 3 Single-Cam Knife + 3 Single-Cam Non-Weapon
#   + 3 Multi-Camera Weapon Pairs + 1 Multi-Camera Non-Weapon Pair
# ==============================================================================
SELECTED_SINGLE_VIDEOS = [
    # --- 1. 3 CCTV HANDGUN SCENES ---
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
        "candidate_name": "CCTV_HG_SR_2026-09-10_214325-HANDGUN_56698fa972b3_",
        "title": "CCTV Handgun 2 — Retail Counter Armed Suspect (214325) [Enhanced]",
        "category": "Handgun",
        "version": "Enhanced",
    },
    {
        "video_id": "HG03_Original",
        "pair_id": "HG03",
        "candidate_name": "CCTV_HG_InStore_Handgun",
        "title": "CCTV Handgun 3 — In-Store Surveillance Handgun [Original]",
        "category": "Handgun",
        "version": "Original",
    },

    # --- 2. 3 CCTV KNIFE SCENES ---
    {
        "video_id": "KN01_Original",
        "pair_id": "KN01",
        "candidate_name": "KN02_220101_StreetKnife_Original",
        "title": "CCTV Knife 1 — Corridor & Street Blade Incident (220101) [Original]",
        "category": "Knife",
        "version": "Original",
    },
    {
        "video_id": "KN01_Enhanced",
        "pair_id": "KN01",
        "candidate_name": "KN02_220101_StreetKnife_Enhanced",
        "title": "CCTV Knife 1 — Corridor & Street Blade Incident (220101) [Enhanced]",
        "category": "Knife",
        "version": "Enhanced",
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

    # --- 3. 3 SINGLE-CAMERA NON-WEAPON SCENES (ALL 0 FP / 0 GHOST BOXES) ---
    {
        "video_id": "NW01",
        "pair_id": "NW01",
        "candidate_name": "NW01_Store_Customer_Interaction_Sample1",
        "title": "Non-Weapon 1 — Convenience Store Customer Interaction (Sample 1)",
        "category": "No-Weapon",
        "version": "Original",
    },
    {
        "video_id": "NW02",
        "pair_id": "NW02",
        "candidate_name": "NW02_Convenience_Store_Counter_Sample2",
        "title": "Non-Weapon 2 — Convenience Store Counter Drinks & Cups (Sample 2)",
        "category": "No-Weapon",
        "version": "Original",
    },
    {
        "video_id": "NW03",
        "pair_id": "NW03",
        "candidate_name": "NW04_CloseUp_EmptyHand_Sample4",
        "title": "Non-Weapon 3 — Indoor Staged Handheld Utensil Distractor (Sample 3)",
        "category": "No-Weapon",
        "version": "Original",
    },
]

# --- 4. 3 MULTI-CAMERA WEAPON PAIRS + 1 MULTI-CAMERA NON-WEAPON PAIR ---
SELECTED_MULTICAM_SCENES = [
    {
        "scene_key": "MC_PAIR01_Enhanced",
        "scene_id": "MC_PAIR01",
        "title": "Multi-Camera Weapon 1 — Dual-Knife Transition Scene 007 (CAM 01 & CAM 02) [Enhanced]",
        "category": "Multi-Camera Knife",
        "version": "Enhanced",
        "cam1_candidate": "STAGED_SCENE007_CAM1_Enhanced",
        "cam2_candidate": "STAGED_SCENE007_CAM2_Enhanced",
    },
    {
        "scene_key": "MC_PAIR02_Enhanced",
        "scene_id": "MC_PAIR02",
        "title": "Multi-Camera Weapon 2 — Handgun Suspect Transition Scene 012 (CAM 01 & CAM 02) [Enhanced]",
        "category": "Multi-Camera Handgun",
        "version": "Enhanced",
        "cam1_candidate": "MC03_SCENE012_Knife_CAM01_Enhanced",
        "cam2_candidate": "MC03_SCENE012_Knife_CAM02_Enhanced",
    },
    {
        "scene_key": "MC_PAIR02_Original",
        "scene_id": "MC_PAIR02",
        "title": "Multi-Camera Weapon 2 — Handgun Suspect Transition Scene 012 (CAM 01 & CAM 02) [Original]",
        "category": "Multi-Camera Handgun",
        "version": "Original",
        "cam1_candidate": "MC03_SCENE012_Knife_CAM01_Original",
        "cam2_candidate": "MC03_SCENE012_Knife_CAM02_Original",
    },
    {
        "scene_key": "MC_PAIR03_Enhanced",
        "scene_id": "MC_PAIR03",
        "title": "Multi-Camera Weapon 3 — Kitchen Knife Transition Scene 001 (CAM 01 & CAM 02) [Enhanced]",
        "category": "Multi-Camera Knife",
        "version": "Enhanced",
        "cam1_candidate": "MC01_SCENE004_Handgun_CAM01_Enhanced",
        "cam2_candidate": "MC01_SCENE004_Handgun_CAM02_Enhanced",
    },
    {
        "scene_key": "MC_PAIR03_Original",
        "scene_id": "MC_PAIR03",
        "title": "Multi-Camera Weapon 3 — Kitchen Knife Transition Scene 001 (CAM 01 & CAM 02) [Original]",
        "category": "Multi-Camera Knife",
        "version": "Original",
        "cam1_candidate": "MC01_SCENE004_Handgun_CAM01_Original",
        "cam2_candidate": "MC01_SCENE004_Handgun_CAM02_Original",
    },
    {
        "scene_key": "MC_NONWEAPON_Original",
        "scene_id": "MC_NONWEAPON",
        "title": "Multi-Camera Non-Weapon — Staged Smartphone Distractor (CAM 01 & CAM 02) [0 FP]",
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
