"""
Build the Pre-Indexed Web Evaluation Library (`web_app/library_data/`) for the FORENSIKADA Web App.

What this script does:
1. Scans `expert_evaluation_videos/` (all Single-Video Handgun/Knife Original & Enhanced pairs,
   all Non-Handgun videos — including any new staged non-handgun videos added later — and all
   Multi-Camera CAM01 + CAM02 synchronized pairs).
2. Runs Model 12 (`best_weapon_detector_twelfth_model.pth`) at `confidence_threshold=0.20` (`analysis_fps=15`)
   so the web app has full telemetry down to 20% for live threshold filtering & CCTV Intelligence toggling.
3. Re-encodes both Original and Annotated videos into compact, browser-compatible H.264 MP4s
   (`libx264`, `yuv420p`, `+faststart`, max width 960px) using `ffmpeg` so they play smoothly in
   Chrome, Edge, Safari, and mobile browsers and stay small enough for free Hugging Face Spaces hosting.
4. Computes SHA-256 cryptographic hashes of every original video file for Forensic PDF Reports.
5. Writes `web_app/library_data/library_index.json` and per-video telemetry JSONs.

To add new videos later (e.g., new staged non-handgun videos from groupmates):
Just drop the `.mp4` files into `expert_evaluation_videos/NO_WEAPON/` and re-run this script!
"""
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import detect_video_with_model
from training.model_12.evaluate_expert_videos_with_ground_truth import (
    SINGLE_VIDEOS_CATALOG,
    NO_WEAPON_VIDEOS_CATALOG,
    MULTI_CAMERA_CATALOG,
)

MODEL_12_PATH = ROOT / "best_weapon_detector_twelfth_model.pth"
EXPERT_LIB_DIR = ROOT / "expert_evaluation_videos"
WEB_DATA_DIR = ROOT / "web_app" / "library_data"
WEB_VIDEOS_DIR = WEB_DATA_DIR / "videos"
WEB_TELEM_DIR = WEB_DATA_DIR / "telemetry"
TMP_RAW_DIR = ROOT / "outputs" / "web_library_build_tmp"


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


def transcode_to_web_h264(src: Path, dst: Path, max_width: int = 960, crf: int = 26):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-crf", str(crf),
        "-preset", "fast",
        "-movflags", "+faststart",
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


def process_single_entry(
    vid_id: str,
    title: str,
    category: str,
    version: str,
    pair_id: str,
    src_path: Path,
    camera_id: str = "CAM-01",
    analysis_fps: int = 15,
) -> dict:
    telem_path = WEB_TELEM_DIR / f"{vid_id}.json"
    orig_web_mp4 = WEB_VIDEOS_DIR / f"{vid_id}_original.mp4"
    annot_web_mp4 = WEB_VIDEOS_DIR / f"{vid_id}_annotated.mp4"

    if telem_path.exists() and orig_web_mp4.exists() and annot_web_mp4.exists():
        print(f"  [cached] {vid_id} already indexed.", flush=True)
        return json.loads(telem_path.read_text(encoding="utf-8"))["entry"]

    print(f"[*] Indexing {vid_id} ({src_path.name}) with Model 12...", flush=True)
    TMP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_annot_mp4 = TMP_RAW_DIR / f"{vid_id}_raw_annot.mp4"
    raw_csv_50 = TMP_RAW_DIR / f"{vid_id}_50pct.csv"
    raw_csv_20 = TMP_RAW_DIR / f"{vid_id}_20pct.csv"

    # Run at 0.50 to produce clean standard annotated video (operational default)
    detect_video_with_model(
        input_path=src_path,
        output_path=raw_annot_mp4,
        model_path=MODEL_12_PATH,
        confidence_threshold=0.50,
        analysis_fps=analysis_fps,
        save_detected_frames=False,
        csv_output_path=raw_csv_50,
        enable_cctv_intelligence=True,
        enable_temporal_consistency=True,
        camera_id=camera_id,
    )

    # Also capture 0.20 exploratory telemetry (without re-writing video) so user can lower slider to 20%
    dummy_mp4 = TMP_RAW_DIR / f"{vid_id}_20pct_tmp.mp4"
    detect_video_with_model(
        input_path=src_path,
        output_path=dummy_mp4,
        model_path=MODEL_12_PATH,
        confidence_threshold=0.20,
        analysis_fps=analysis_fps,
        save_detected_frames=False,
        csv_output_path=raw_csv_20,
        enable_cctv_intelligence=True,
        enable_temporal_consistency=True,
        camera_id=camera_id,
    )
    if dummy_mp4.exists():
        dummy_mp4.unlink()

    transcode_to_web_h264(src_path, orig_web_mp4, max_width=960, crf=26)
    transcode_to_web_h264(raw_annot_mp4, annot_web_mp4, max_width=960, crf=26)

    orig_meta = get_video_meta(src_path)
    web_meta = get_video_meta(orig_web_mp4)
    file_hash = sha256_file(src_path)

    recs_50 = parse_detections_csv(raw_csv_50, orig_meta["width"], orig_meta["height"])
    recs_20 = parse_detections_csv(raw_csv_20, orig_meta["width"], orig_meta["height"])

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
        "model_name": "Model 12 (Production Video SOTA)",
        "model_checkpoint": MODEL_12_PATH.name,
        "records_50pct": recs_50,
        "records_20pct": recs_20,
    }
    WEB_TELEM_DIR.mkdir(parents=True, exist_ok=True)
    telem_path.write_text(json.dumps(telem_payload, indent=2), encoding="utf-8")
    return entry


def main():
    WEB_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_TELEM_DIR.mkdir(parents=True, exist_ok=True)

    single_entries = []
    # 1. Single-Video Weapon Pairs (Original & Enhanced)
    for item in SINGLE_VIDEOS_CATALOG:
        pid = item["pair_id"]
        for ver_tag, key in [("Original", "original_path"), ("Enhanced", "enhanced_path")]:
            src_p = item[key]
            if not src_p.exists():
                continue
            vid_id = f"{pid}_{ver_tag}"
            entry = process_single_entry(
                vid_id=vid_id,
                title=f"{item['title']} [{ver_tag}]",
                category=item["category"],
                version=ver_tag,
                pair_id=pid,
                src_path=src_p,
                camera_id=pid,
                analysis_fps=15,
            )
            single_entries.append(entry)

    # 2. Non-Handgun / No-Weapon Videos (Catalog + any extra .mp4 dropped into expert_evaluation_videos/NO_WEAPON/)
    seen_nw_ids = set()
    for item in NO_WEAPON_VIDEOS_CATALOG:
        pid = item["pair_id"]
        src_p = item["video_path"]
        if not src_p.exists():
            continue
        seen_nw_ids.add(pid)
        entry = process_single_entry(
            vid_id=pid,
            title=item["title"],
            category="No-Weapon",
            version="Original",
            pair_id=pid,
            src_path=src_p,
            camera_id=pid,
            analysis_fps=15,
        )
        single_entries.append(entry)

    nw_folder = EXPERT_LIB_DIR / "NO_WEAPON"
    if nw_folder.exists():
        for extra_mp4 in sorted(nw_folder.glob("*.mp4")):
            stem = extra_mp4.stem
            if stem not in seen_nw_ids:
                entry = process_single_entry(
                    vid_id=stem,
                    title=f"Non-Handgun — {stem}",
                    category="No-Weapon",
                    version="Original",
                    pair_id=stem,
                    src_path=extra_mp4,
                    camera_id=stem,
                    analysis_fps=15,
                )
                single_entries.append(entry)

    # 3. Multi-Camera Synchronized Pairs (CAM 01 & CAM 02, Original & Enhanced)
    multicam_scenes = []
    for item in MULTI_CAMERA_CATALOG:
        sid = item["scene_id"]
        for ver_tag, k1, k2 in [
            ("Original", "cam1_original", "cam2_original"),
            ("Enhanced", "cam1_enhanced", "cam2_enhanced"),
        ]:
            p1, p2 = item[k1], item[k2]
            if not (p1.exists() and p2.exists()):
                continue
            e1 = process_single_entry(
                vid_id=f"{sid}_CAM01_{ver_tag}",
                title=f"{item['title']} — CAM-01 [{ver_tag}]",
                category=item["category"],
                version=ver_tag,
                pair_id=sid,
                src_path=p1,
                camera_id="CAM-01",
                analysis_fps=15,
            )
            e2 = process_single_entry(
                vid_id=f"{sid}_CAM02_{ver_tag}",
                title=f"{item['title']} — CAM-02 [{ver_tag}]",
                category=item["category"],
                version=ver_tag,
                pair_id=sid,
                src_path=p2,
                camera_id="CAM-02",
                analysis_fps=15,
            )
            multicam_scenes.append({
                "scene_key": f"{sid}_{ver_tag}",
                "scene_id": sid,
                "title": f"{item['title']} [{ver_tag}]",
                "category": item["category"],
                "version": ver_tag,
                "cam1": e1,
                "cam2": e2,
            })

    index_payload = {
        "system_name": "FORENSIKADA",
        "default_model": "Model 12 (Production Video SOTA)",
        "single_videos": single_entries,
        "multicam_scenes": multicam_scenes,
    }
    (WEB_DATA_DIR / "library_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in WEB_VIDEOS_DIR.glob("*.mp4")) / (1024 * 1024)
    print(f"\n[+] Web Library Ready! {len(single_entries)} single videos + {len(multicam_scenes)} multi-camera scenes.")
    print(f"[+] Total H.264 video bundle size: {total_mb:.1f} MB -> {WEB_DATA_DIR}")


if __name__ == "__main__":
    main()

