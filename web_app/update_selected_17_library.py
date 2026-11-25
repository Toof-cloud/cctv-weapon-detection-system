"""
Build the Clean Unified FORENSIKADA Web Library (`web_app/library_data/`):
1. Annotates the 4 missing unenhanced Original counterparts (`214325`, `220510`, `CAM1_SCENE007`, `CAM2_SCENE007`)
   with Model 12 so EVERY weapon video (3 Handgun, 3 Knife, 3 Multi-Camera Weapon) has BOTH
   its true Unenhanced Original AND its BasicVSR++ Enhanced version.
2. Stores ONE clean entry per scene in `library_index.json`:
   - 9 Single-Camera entries (3 Handgun + 3 Knife + 3 Non-Weapon) — no duplicate [Original]/[Enhanced] clutter!
   - 4 Multi-Camera entries (3 Weapon Pairs + 1 Non-Weapon Pair) — no duplicate [Original]/[Enhanced] clutter!
   Each entry bundles both Original (raw + annotated + telemetry) and Enhanced (raw + annotated + telemetry)
   so clicking "Enhance video" in the UI immediately switches to the Enhanced Annotated video!
3. Wall-clock synchronizes the Multi-Camera H.264 streams (trimming the 1.0s Tapo clock start offset on
   CAM1_SCENE007 [11:59:03 -> 11:59:04] and Smartphone CAM02 [12:49:15 -> 12:49:16], while SCENE012 [12:05:22]
   and SCENE001 [11:48:00] already start at 0.0s offset) and shifts CSV timestamps accordingly so both POVs
   play in exact wall-clock sync!
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

import run_full_pipeline
from app.services.detection_service import DetectionService

MODEL_12 = ROOT / "best_weapon_detector_twelfth_model.pth"
DRIVE_ROOT = ROOT / "outputs" / "00_GOOGLE_DRIVE_MODEL12_ANNOTATED_VIDEOS_FOR_SELECTION"
EXTRA_ORIG_CACHE = ROOT / "outputs" / "01_final_model12_annotated_library" / "EXTRA_ORIGINALS_M12"
WEB_DATA_DIR = ROOT / "web_app" / "library_data"
WEB_VIDEOS_DIR = WEB_DATA_DIR / "videos"
WEB_TELEM_DIR = WEB_DATA_DIR / "telemetry"

_CACHED_DET = None


def _get_det(confidence_threshold: float = 0.50, model_path=MODEL_12):
    global _CACHED_DET
    if _CACHED_DET is None:
        _CACHED_DET = DetectionService(confidence_threshold=confidence_threshold, model_path=model_path)
    else:
        _CACHED_DET.confidence_threshold = confidence_threshold
    return _CACHED_DET


run_full_pipeline.DetectionService = _get_det


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


def ensure_annotated(tag: str, src_path: Path, existing_mp4: Path | None = None, existing_csv: Path | None = None) -> tuple[Path, Path]:
    if existing_mp4 and existing_csv and existing_mp4.exists() and existing_csv.exists():
        return existing_mp4, existing_csv
    EXTRA_ORIG_CACHE.mkdir(parents=True, exist_ok=True)
    out_mp4 = EXTRA_ORIG_CACHE / f"{tag}_M12_annotated.mp4"
    out_csv = EXTRA_ORIG_CACHE / f"{tag}_M12_detections.csv"
    if not (out_mp4.exists() and out_csv.exists()):
        print(f"[*] Running Model 12 on Original counterpart: {tag} ({src_path.name})...", flush=True)
        run_full_pipeline.detect_video_with_model(
            input_path=src_path,
            output_path=out_mp4,
            model_path=MODEL_12,
            confidence_threshold=0.50,
            analysis_fps=15,
            save_detected_frames=False,
            csv_output_path=out_csv,
            enable_cctv_intelligence=True,
            enable_temporal_consistency=True,
            camera_id=tag[:20],
        )
    return out_mp4, out_csv


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


def transcode_to_web_h264(src: Path, dst: Path, max_width: int = 960, crf: int = 26, trim_start_sec: float = 0.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG_BIN, "-y"]
    if trim_start_sec > 0.0:
        cmd += ["-ss", f"{trim_start_sec:.3f}"]
    cmd += [
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


def _fmt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    m = int(sec // 60)
    s = sec - m * 60
    return f"00:{m:02d}:{s:05.2f}"


def parse_detections_csv(csv_path: Path, orig_w: int, orig_h: int, trim_start_sec: float = 0.0, fps: float = 15.0) -> list[dict]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    records = []
    trim_frames = int(round(trim_start_sec * fps)) if trim_start_sec > 0 else 0
    for _, r in df.iterrows():
        ts_raw = float(r["timestamp_seconds"])
        if ts_raw < trim_start_sec - 0.02:
            continue
        ts_adj = max(0.0, round(ts_raw - trim_start_sec, 3))
        fn_adj = max(1, int(r["frame_number"]) - trim_frames)
        try:
            bx = ast.literal_eval(str(r["bounding_box"]))
            bx = [int(v) for v in bx]
        except Exception:
            bx = [0, 0, 0, 0]
        rej = str(r["rejection_reason"]).strip() if pd.notna(r["rejection_reason"]) else ""
        if rej.lower() == "nan":
            rej = ""
        records.append({
            "frame_number": fn_adj,
            "timestamp_seconds": ts_adj,
            "timestamp_formatted": _fmt_ts(ts_adj) if trim_start_sec > 0 else str(r["timestamp_formatted"]),
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


def package_stream(
    vid_id: str,
    title: str,
    category: str,
    camera_id: str,
    orig_src: Path,
    orig_annot: Path,
    orig_csv: Path,
    enh_src: Path | None = None,
    enh_annot: Path | None = None,
    enh_csv: Path | None = None,
    trim_start_sec: float = 0.0,
    offset_seconds: float = 0.0,
) -> dict:
    print(f"  -> Packaging {vid_id} (has_enhanced={enh_src is not None}, trim={trim_start_sec}s, offset={offset_seconds}s)", flush=True)
    orig_raw_web = WEB_VIDEOS_DIR / f"{vid_id}_orig_raw.mp4"
    orig_ann_web = WEB_VIDEOS_DIR / f"{vid_id}_orig_annot.mp4"
    transcode_to_web_h264(orig_src, orig_raw_web, trim_start_sec=trim_start_sec)
    transcode_to_web_h264(orig_annot, orig_ann_web, trim_start_sec=trim_start_sec)

    om = get_video_meta(orig_src)
    wm = get_video_meta(orig_raw_web)
    fhash = sha256_file(orig_src)
    recs_orig = parse_detections_csv(orig_csv, om["width"], om["height"], trim_start_sec=trim_start_sec, fps=om["fps"])

    orig_telem_rel = f"telemetry/{vid_id}_orig.json"
    (WEB_TELEM_DIR / f"{vid_id}_orig.json").write_text(
        json.dumps({"records_50pct": recs_orig, "records_20pct": recs_orig}, indent=2),
        encoding="utf-8",
    )

    has_enh = enh_src is not None and enh_annot is not None and enh_csv is not None
    enh_raw_rel = None
    enh_ann_rel = None
    enh_telem_rel = None
    enh_hash = None
    enh_filename = None

    if has_enh:
        enh_raw_web = WEB_VIDEOS_DIR / f"{vid_id}_enh_raw.mp4"
        enh_ann_web = WEB_VIDEOS_DIR / f"{vid_id}_enh_annot.mp4"
        transcode_to_web_h264(enh_src, enh_raw_web, trim_start_sec=trim_start_sec)
        transcode_to_web_h264(enh_annot, enh_ann_web, trim_start_sec=trim_start_sec)
        em = get_video_meta(enh_src)
        recs_enh = parse_detections_csv(enh_csv, em["width"], em["height"], trim_start_sec=trim_start_sec, fps=em["fps"])
        enh_telem_rel = f"telemetry/{vid_id}_enh.json"
        (WEB_TELEM_DIR / f"{vid_id}_enh.json").write_text(
            json.dumps({"records_50pct": recs_enh, "records_20pct": recs_enh}, indent=2),
            encoding="utf-8",
        )
        enh_raw_rel = f"videos/{enh_raw_web.name}"
        enh_ann_rel = f"videos/{enh_ann_web.name}"
        enh_hash = sha256_file(enh_src)
        enh_filename = enh_src.name

    return {
        "video_id": vid_id,
        "pair_id": vid_id,
        "title": title,
        "category": category,
        "version": "Original",
        "has_enhanced": has_enh,
        "camera_id": camera_id,
        "offset_seconds": round(float(offset_seconds), 3),
        "source_filename": orig_src.name,
        "enhanced_filename": enh_filename,
        "sha256": fhash,
        "enhanced_sha256": enh_hash,
        "orig_meta": om,
        "web_meta": wm,
        "original_video_rel": f"videos/{orig_raw_web.name}",
        "annotated_video_rel": f"videos/{orig_ann_web.name}",
        "telemetry_rel": orig_telem_rel,
        "enhanced_original_video_rel": enh_raw_rel,
        "enhanced_annotated_video_rel": enh_ann_rel,
        "enhanced_telemetry_rel": enh_telem_rel,
    }


def main() -> None:
    e_dir = ROOT / "expert_evaluation_videos"
    s_dir = ROOT / "samples"
    d_hg = DRIVE_ROOT / "01_CCTV_HANDGUN_CANDIDATES"
    d_kn = DRIVE_ROOT / "02_CCTV_KNIFE_CANDIDATES"
    d_mc = DRIVE_ROOT / "03_MULTI_CAMERA_STAGED_WEAPON_CANDIDATES"
    d_nw_st = DRIVE_ROOT / "04_NON_HANDGUN_STAGED_DATASET_CANDIDATES"
    d_nw_cs = DRIVE_ROOT / "05_NON_HANDGUN_CONVENIENCE_STORE_CANDIDATES"

    # Annotate the 4 missing Original counterparts so all weapon clips have true Original + BasicVSR++ Enhanced!
    hg214325_orig_src = s_dir / "05_historical_consultations" / "DEFENSE" / "ORIGINAL VIDEOS" / "Screen Recording 2026-09-10 214325-HANDGUN.mp4"
    hg214325_orig_mp4, hg214325_orig_csv = ensure_annotated("HG02_214325_Original", hg214325_orig_src)

    kn220510_orig_src = s_dir / "05_historical_consultations" / "DEFENSE" / "ORIGINAL VIDEOS" / "Screen Recording 2026-09-10 220510-KNIFE.mp4"
    kn220510_orig_mp4, kn220510_orig_csv = ensure_annotated("KN02_220510_Original", kn220510_orig_src)

    sc007_c1_orig_src = s_dir / "05_historical_consultations" / "To_enhance" / "CAM1_SCENE007.mov"
    sc007_c1_orig_mp4, sc007_c1_orig_csv = ensure_annotated("SCENE007_CAM01_Original", sc007_c1_orig_src)

    sc007_c2_orig_src = s_dir / "05_historical_consultations" / "To_enhance" / "CAM2_SCENE007.mp4"
    sc007_c2_orig_mp4, sc007_c2_orig_csv = ensure_annotated("SCENE007_CAM02_Original", sc007_c2_orig_src)

    if WEB_VIDEOS_DIR.exists():
        shutil.rmtree(WEB_VIDEOS_DIR)
    if WEB_TELEM_DIR.exists():
        shutil.rmtree(WEB_TELEM_DIR)
    WEB_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_TELEM_DIR.mkdir(parents=True, exist_ok=True)

    # ==================== 9 SINGLE-VIDEO ENTRIES (ONE PER SCENE) ====================
    single_entries = [
        # 1. 3 CCTV Handgun Videos
        package_stream(
            vid_id="HG01",
            title="CCTV Handgun 1 — Store Counter Robbery (230249)",
            category="Handgun",
            camera_id="HG01",
            orig_src=e_dir / "HANDGUN" / "HG01_Original.mp4",
            orig_annot=d_hg / "HG01_230249_StoreRobbery_Original_M12_annotated.mp4",
            orig_csv=d_hg / "HG01_230249_StoreRobbery_Original_M12_detections.csv",
            enh_src=e_dir / "HANDGUN" / "HG01_Enhanced.mp4",
            enh_annot=d_hg / "HG01_230249_StoreRobbery_Enhanced_M12_annotated.mp4",
            enh_csv=d_hg / "HG01_230249_StoreRobbery_Enhanced_M12_detections.csv",
        ),
        package_stream(
            vid_id="HG02",
            title="CCTV Handgun 2 — Retail Counter Armed Suspect (214325)",
            category="Handgun",
            camera_id="HG02",
            orig_src=hg214325_orig_src,
            orig_annot=hg214325_orig_mp4,
            orig_csv=hg214325_orig_csv,
            enh_src=s_dir / "05_historical_consultations" / "DEFENSE" / "HANDGUN VIDEOS" / "Screen Recording 2026-09-10 214325-HANDGUN_56698fa972b3_enhanced.mp4",
            enh_annot=d_hg / "CCTV_HG_SR_2026-09-10_214325-HANDGUN_56698fa972b3__M12_annotated.mp4",
            enh_csv=d_hg / "CCTV_HG_SR_2026-09-10_214325-HANDGUN_56698fa972b3__M12_detections.csv",
        ),
        package_stream(
            vid_id="HG03",
            title="CCTV Handgun 3 — In-Store Surveillance Handgun (213340)",
            category="Handgun",
            camera_id="HG03",
            orig_src=e_dir / "HANDGUN" / "HG02_Original.mp4",
            orig_annot=d_hg / "HG02_213340_StreetHandgun_Original_M12_annotated.mp4",
            orig_csv=d_hg / "HG02_213340_StreetHandgun_Original_M12_detections.csv",
            enh_src=e_dir / "HANDGUN" / "HG02_Enhanced.mp4",
            enh_annot=d_hg / "HG02_213340_StreetHandgun_Enhanced_M12_annotated.mp4",
            enh_csv=d_hg / "HG02_213340_StreetHandgun_Enhanced_M12_detections.csv",
        ),

        # 2. 3 CCTV Knife Videos
        package_stream(
            vid_id="KN01",
            title="CCTV Knife 1 — Corridor & Street Blade Incident (220101)",
            category="Knife",
            camera_id="KN01",
            orig_src=e_dir / "KNIFE" / "KN02_Original.mp4",
            orig_annot=d_kn / "KN02_220101_StreetKnife_Original_M12_annotated.mp4",
            orig_csv=d_kn / "KN02_220101_StreetKnife_Original_M12_detections.csv",
            enh_src=e_dir / "KNIFE" / "KN02_Enhanced.mp4",
            enh_annot=d_kn / "KN02_220101_StreetKnife_Enhanced_M12_annotated.mp4",
            enh_csv=d_kn / "KN02_220101_StreetKnife_Enhanced_M12_detections.csv",
        ),
        package_stream(
            vid_id="KN02",
            title="CCTV Knife 2 — Close-Quarters Blade Threat (220510)",
            category="Knife",
            camera_id="KN02",
            orig_src=kn220510_orig_src,
            orig_annot=kn220510_orig_mp4,
            orig_csv=kn220510_orig_csv,
            enh_src=s_dir / "05_historical_consultations" / "DEFENSE" / "KNIFE VIDEOS" / "Screen Recording 2026-09-10 220510-KNIFE_de569e413ac3_enhanced.mp4",
            enh_annot=d_kn / "CCTV_KN_SR_2026-09-10_220510-KNIFE_de569e413ac3_en_M12_annotated.mp4",
            enh_csv=d_kn / "CCTV_KN_SR_2026-09-10_220510-KNIFE_de569e413ac3_en_M12_detections.csv",
        ),
        package_stream(
            vid_id="KN03",
            title="CCTV Knife 3 — Fast Indoor Blade Swing (230047)",
            category="Knife",
            camera_id="KN03",
            orig_src=e_dir / "KNIFE" / "KN01_Original.mp4",
            orig_annot=d_kn / "KN01_230047_FastKnifeSwing_Original_M12_annotated.mp4",
            orig_csv=d_kn / "KN01_230047_FastKnifeSwing_Original_M12_detections.csv",
            enh_src=e_dir / "KNIFE" / "KN01_Enhanced.mp4",
            enh_annot=d_kn / "KN01_230047_FastKnifeSwing_Enhanced_M12_annotated.mp4",
            enh_csv=d_kn / "KN01_230047_FastKnifeSwing_Enhanced_M12_detections.csv",
        ),

        # 3. 3 Single-Cam Non-Weapon Videos (0 FP)
        package_stream(
            vid_id="NW01",
            title="Non-Weapon 1 — Convenience Store Customer Interaction (Sample 1)",
            category="No-Weapon",
            camera_id="NW01",
            orig_src=e_dir / "NO_WEAPON" / "NW01.mp4",
            orig_annot=d_nw_cs / "NW01_Store_Customer_Interaction_Sample1_M12_annotated.mp4",
            orig_csv=d_nw_cs / "NW01_Store_Customer_Interaction_Sample1_M12_detections.csv",
        ),
        package_stream(
            vid_id="NW02",
            title="Non-Weapon 2 — Convenience Store Counter Drinks & Cups (Sample 2)",
            category="No-Weapon",
            camera_id="NW02",
            orig_src=e_dir / "NO_WEAPON" / "NW02.mp4",
            orig_annot=d_nw_cs / "NW02_Convenience_Store_Counter_Sample2_M12_annotated.mp4",
            orig_csv=d_nw_cs / "NW02_Convenience_Store_Counter_Sample2_M12_detections.csv",
        ),
        package_stream(
            vid_id="NW03",
            title="Non-Weapon 3 — Indoor Staged Handheld Utensil Distractor (Sample 3)",
            category="No-Weapon",
            camera_id="NW03",
            orig_src=e_dir / "NO_WEAPON" / "NW04.mp4",
            orig_annot=d_nw_cs / "NW04_CloseUp_EmptyHand_Sample4_M12_annotated.mp4",
            orig_csv=d_nw_cs / "NW04_CloseUp_EmptyHand_Sample4_M12_detections.csv",
        ),
    ]

    # ==================== 4 MULTI-CAMERA ENTRIES (UNTRIMMED + EXACT ANALYST START OFFSETS) ====================
    # Session time = video_time + offset_seconds (matches desktop mockup_ui/multi_camera_panel.py 1:1!)
    nov20_dir = s_dir / "03_staged_and_multicamera" / "nov20_staged_test_videos" / "Nov 20 - Staged Test Videos"
    st_enh_dir = s_dir / "05_historical_consultations" / "DEFENSE" / "STAGED DATASET"

    mc_pair1_c1 = package_stream(
        vid_id="MC01_CAM01",
        title="Multi-Camera Weapon 1 (Scene 007) — CAM-01",
        category="Multi-Camera Handgun & Knife",
        camera_id="CAM-01",
        orig_src=sc007_c1_orig_src,
        orig_annot=sc007_c1_orig_mp4,
        orig_csv=sc007_c1_orig_csv,
        enh_src=st_enh_dir / "CAM1_SCENE007_enhanced.mp4",
        enh_annot=d_mc / "STAGED_SCENE007_CAM1_Enhanced_M12_annotated.mp4",
        enh_csv=d_mc / "STAGED_SCENE007_CAM1_Enhanced_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=0.000,
    )
    mc_pair1_c2 = package_stream(
        vid_id="MC01_CAM02",
        title="Multi-Camera Weapon 1 (Scene 007) — CAM-02",
        category="Multi-Camera Handgun & Knife",
        camera_id="CAM-02",
        orig_src=sc007_c2_orig_src,
        orig_annot=sc007_c2_orig_mp4,
        orig_csv=sc007_c2_orig_csv,
        enh_src=st_enh_dir / "CAM2_SCENE007_enhanced.mp4",
        enh_annot=d_mc / "STAGED_SCENE007_CAM2_Enhanced_M12_annotated.mp4",
        enh_csv=d_mc / "STAGED_SCENE007_CAM2_Enhanced_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=1.780,  # Analyst-arranged exact start offset (+1.780 s)
    )

    mc_pair2_c1 = package_stream(
        vid_id="MC02_CAM01",
        title="Multi-Camera Weapon 2 (Scene 012) — CAM-01",
        category="Multi-Camera Handgun",
        camera_id="CAM-01",
        orig_src=e_dir / "MULTI_CAMERA" / "MC03_CAM01_Original.mov",
        orig_annot=d_mc / "MC03_SCENE012_Knife_CAM01_Original_M12_annotated.mp4",
        orig_csv=d_mc / "MC03_SCENE012_Knife_CAM01_Original_M12_detections.csv",
        enh_src=e_dir / "MULTI_CAMERA" / "MC03_CAM01_Enhanced.mp4",
        enh_annot=d_mc / "MC03_SCENE012_Knife_CAM01_Enhanced_M12_annotated.mp4",
        enh_csv=d_mc / "MC03_SCENE012_Knife_CAM01_Enhanced_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=0.000,
    )
    mc_pair2_c2 = package_stream(
        vid_id="MC02_CAM02",
        title="Multi-Camera Weapon 2 (Scene 012) — CAM-02",
        category="Multi-Camera Handgun",
        camera_id="CAM-02",
        orig_src=e_dir / "MULTI_CAMERA" / "MC03_CAM02_Original.mp4",
        orig_annot=d_mc / "MC03_SCENE012_Knife_CAM02_Original_M12_annotated.mp4",
        orig_csv=d_mc / "MC03_SCENE012_Knife_CAM02_Original_M12_detections.csv",
        enh_src=e_dir / "MULTI_CAMERA" / "MC03_CAM02_Enhanced.mp4",
        enh_annot=d_mc / "MC03_SCENE012_Knife_CAM02_Enhanced_M12_annotated.mp4",
        enh_csv=d_mc / "MC03_SCENE012_Knife_CAM02_Enhanced_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=0.760,  # Analyst-arranged exact start offset (+0.760 s)
    )

    mc_pair3_c1 = package_stream(
        vid_id="MC03_CAM01",
        title="Multi-Camera Weapon 3 (Scene 001) — CAM-01",
        category="Multi-Camera Knife",
        camera_id="CAM-01",
        orig_src=e_dir / "MULTI_CAMERA" / "MC01_CAM01_Original.mov",
        orig_annot=d_mc / "MC01_SCENE004_Handgun_CAM01_Original_M12_annotated.mp4",
        orig_csv=d_mc / "MC01_SCENE004_Handgun_CAM01_Original_M12_detections.csv",
        enh_src=e_dir / "MULTI_CAMERA" / "MC01_CAM01_Enhanced.mp4",
        enh_annot=d_mc / "MC01_SCENE004_Handgun_CAM01_Enhanced_M12_annotated.mp4",
        enh_csv=d_mc / "MC01_SCENE004_Handgun_CAM01_Enhanced_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=0.000,
    )
    mc_pair3_c2 = package_stream(
        vid_id="MC03_CAM02",
        title="Multi-Camera Weapon 3 (Scene 001) — CAM-02",
        category="Multi-Camera Knife",
        camera_id="CAM-02",
        orig_src=e_dir / "MULTI_CAMERA" / "MC01_CAM02_Original.mp4",
        orig_annot=d_mc / "MC01_SCENE004_Handgun_CAM02_Original_M12_annotated.mp4",
        orig_csv=d_mc / "MC01_SCENE004_Handgun_CAM02_Original_M12_detections.csv",
        enh_src=e_dir / "MULTI_CAMERA" / "MC01_CAM02_Enhanced.mp4",
        enh_annot=d_mc / "MC01_SCENE004_Handgun_CAM02_Enhanced_M12_annotated.mp4",
        enh_csv=d_mc / "MC01_SCENE004_Handgun_CAM02_Enhanced_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=1.100,  # Analyst-arranged exact start offset (+1.100 s)
    )

    mc_pair4_c1 = package_stream(
        vid_id="MC04_CAM01",
        title="Multi-Camera Non-Weapon (Smartphone) — CAM-01",
        category="Multi-Camera No-Weapon",
        camera_id="CAM-01",
        orig_src=nov20_dir / "CAM 01" / "1869451F17C5_20261120124918870_cam0.mp4",
        orig_annot=d_nw_st / "NOV20_Pair05_Smartphone_CAM01_M12_annotated.mp4",
        orig_csv=d_nw_st / "NOV20_Pair05_Smartphone_CAM01_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=0.000,
    )
    mc_pair4_c2 = package_stream(
        vid_id="MC04_CAM02",
        title="Multi-Camera Non-Weapon (Smartphone) — CAM-02",
        category="Multi-Camera No-Weapon",
        camera_id="CAM-02",
        orig_src=nov20_dir / "CAM 02" / "1795150158464_0.mp4",
        orig_annot=d_nw_st / "NOV20_Pair05_Smartphone_CAM02_M12_annotated.mp4",
        orig_csv=d_nw_st / "NOV20_Pair05_Smartphone_CAM02_M12_detections.csv",
        trim_start_sec=0.0,
        offset_seconds=0.000,
    )

    multicam_scenes = [
        {
            "scene_key": "MC01",
            "scene_id": "SCENE-007",
            "title": "Multi-Camera Weapon 1 — Handgun & Knife Transition Scene 007 (CAM 01 & CAM 02)",
            "category": "Multi-Camera Handgun & Knife",
            "has_enhanced": True,
            "matching_window_seconds": 1.5,
            "sync_clock": "CAM-01 offset 0.000 s · CAM-02 offset +1.780 s",
            "cam1": mc_pair1_c1,
            "cam2": mc_pair1_c2,
        },
        {
            "scene_key": "MC02",
            "scene_id": "SCENE-012",
            "title": "Multi-Camera Weapon 2 — Handgun Suspect Transition Scene 012 (CAM 01 & CAM 02)",
            "category": "Multi-Camera Handgun",
            "has_enhanced": True,
            "matching_window_seconds": 1.5,
            "sync_clock": "CAM-01 offset 0.000 s · CAM-02 offset +0.760 s",
            "cam1": mc_pair2_c1,
            "cam2": mc_pair2_c2,
        },
        {
            "scene_key": "MC03",
            "scene_id": "SCENE-001",
            "title": "Multi-Camera Weapon 3 — Kitchen Knife Transition Scene 001 (CAM 01 & CAM 02)",
            "category": "Multi-Camera Knife",
            "has_enhanced": True,
            "matching_window_seconds": 1.5,
            "sync_clock": "CAM-01 offset 0.000 s · CAM-02 offset +1.100 s",
            "cam1": mc_pair3_c1,
            "cam2": mc_pair3_c2,
        },
        {
            "scene_key": "MC04",
            "scene_id": "PAIR-05",
            "title": "Multi-Camera Non-Weapon — Staged Smartphone Distractor (CAM 01 & CAM 02)",
            "category": "Multi-Camera No-Weapon",
            "has_enhanced": False,
            "matching_window_seconds": 1.5,
            "sync_clock": "CAM-01 offset 0.000 s · CAM-02 offset 0.000 s",
            "cam1": mc_pair4_c1,
            "cam2": mc_pair4_c2,
        },
    ]

    index_payload = {
        "system_name": "FORENSIKADA",
        "default_model": "Model 12 (Final Production SOTA)",
        "single_videos": single_entries,
        "multicam_scenes": multicam_scenes,
    }
    (WEB_DATA_DIR / "library_index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in WEB_VIDEOS_DIR.glob("*.mp4")) / (1024 * 1024)
    print(f"\n[DONE] Built Unified Synced Library: {len(single_entries)} single clips + {len(multicam_scenes)} multi-cam scenes ({total_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
