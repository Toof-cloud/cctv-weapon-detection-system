"""
FORENSIKADA — Web-Based Multi-Camera CCTV Forensic Weapon Detection System.

1:1 visual and functional web adaptation of the FORENSIKADA Desktop Application (`mockup_ui/app.py`):
- Exact dark navy header (#0d1c42) with official Forensikada logo and 5 interactive action buttons:
  Import videos | Close videos | Forensic report | Save video + report | Settings (⚙)
- Exact mode tab strip (#edf2fa) with rounded navy active pill: Single video | Multi-camera | Evaluator guide
- Exact 3-column DetectionPageShell layout (sidePanel | workspace | summaryPanel):
  * Left: EVIDENCE SOURCE card, Enhancement (BasicVSR++ one-click toggle), VIEW MODE segmented toggle
    (Original | Detected), CONFIDENCE THRESHOLD navy slider, Analyze video / Observation review buttons,
    ANALYSIS STATUS card, and TRAINED MODEL footer.
  * Center: VIDEO WEAPON DETECTION heading, dark cinema video player, clickable DETECTED OBJECTS table
    (Time | Object | Confidence | Frame), and safety/metadata footer.
  * Right: DETECTION SUMMARY with 3 metric cards (DETECTIONS | HANDGUNS | KNIVES), CURRENT FRAME preview
    with green confidence pill, and observation count note.
- Interactive desktop-style dialogs (@st.dialog) for Import Videos (Library + Live Custom Upload <=25MB/<=15s),
  Observation Review (Accept / Reject / Uncertain + notes), Forensic Report (in-app preview + PDF/CSV download),
  and System Settings (Light / Dark theme + CCTV Intelligence toggle).
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web_app.pdf_export_helper import build_multicam_pdf_bytes, build_single_video_pdf_bytes

WEB_DATA_DIR = ROOT / "web_app" / "library_data"
INDEX_PATH = WEB_DATA_DIR / "library_index.json"
MODEL_12_PATH = ROOT / "best_weapon_detector_twelfth_model.pth"
ASSETS_DIR = ROOT / "web_app" / "assets"
LOGO_PATH = ASSETS_DIR / "forensikada-logo.png"
ICON_PATH = ASSETS_DIR / "forensikada-app-icon.png"

MAX_UPLOAD_MB = 25.0
MAX_UPLOAD_SECONDS = 15.0
GOOGLE_FORM_URL = ""  # Optional: paste your Google Forms evaluation link here

st.set_page_config(
    page_title="Forensikada · Video Weapon Detection",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    return ""


@st.cache_data(show_spinner=False)
def load_library_index() -> dict:
    if not INDEX_PATH.exists():
        return {"single_videos": [], "multicam_scenes": []}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_telemetry(telem_rel: str) -> dict:
    p = WEB_DATA_DIR / telem_rel
    if not p.exists():
        return {"records_50pct": [], "records_20pct": []}
    return json.loads(p.read_text(encoding="utf-8"))


def init_state(single_videos: list[dict], mc_scenes: list[dict]) -> None:
    ss = st.session_state
    ss.setdefault("theme", "light")
    ss.setdefault("active_mode", "Single video")
    ss.setdefault("view_mode", "Original")
    ss.setdefault("conf_pct", 50)
    ss.setdefault("cctv_intel", True)
    ss.setdefault("reviews", {})
    ss.setdefault("selected_obs_idx", 0)
    ss.setdefault("custom_result", None)
    ss.setdefault("active_video_id", None)
    ss.setdefault("analysis_ready", False)
    ss.setdefault("splash_shown", False)
    ss.setdefault("active_mc_key", mc_scenes[0]["scene_key"] if mc_scenes else None)
    ss.setdefault("mc_view_mode", "Detected")
    ss.setdefault("mc_conf_pct", 50)


def filter_records(telem: dict, threshold: float, cctv_intel_enabled: bool) -> list[dict]:
    base = telem.get("records_50pct", []) if threshold >= 0.50 else telem.get("records_20pct", [])
    out = []
    for r in base:
        if r["confidence_score"] < threshold:
            continue
        if cctv_intel_enabled:
            if r["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL") and not r["rejection_reason"]:
                out.append(r)
        else:
            out.append(r)
    return out


def extract_annotated_frame_rgb(video_path: Path, frame_number: int, norm_box: list[float], label: str, conf: float):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 1))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    x1 = int(round(norm_box[0] * w))
    y1 = int(round(norm_box[1] * h))
    x2 = int(round(norm_box[2] * w))
    y2 = int(round(norm_box[3] * h))
    color = (0, 60, 255) if label == "handgun" else (0, 165, 255)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
    tag = f"{label.upper()} {conf*100:.1f}%"
    cv2.rectangle(frame_bgr, (x1, max(0, y1 - 26)), (min(w, x1 + 210), y1), color, -1)
    cv2.putText(frame_bgr, tag, (x1 + 5, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def run_live_custom_upload(uploaded_file, threshold: float, cctv_intel_enabled: bool):
    from run_full_pipeline import detect_video_with_model
    import torch

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"Uploaded file is {size_mb:.1f} MB, which exceeds the {MAX_UPLOAD_MB:.0f} MB web upload limit.")
        return None

    tmp_dir = Path(tempfile.mkdtemp(prefix="forensikada_upload_"))
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    in_path = tmp_dir / f"input{suffix}"
    in_path.write_bytes(uploaded_file.getvalue())

    cap = cv2.VideoCapture(str(in_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_frames <= 0 or w <= 0 or h <= 0:
        st.error("Could not decode the uploaded video file. Please upload a standard MP4/AVI/MOV video.")
        return None

    duration = n_frames / max(1.0, fps)
    if duration > MAX_UPLOAD_SECONDS:
        st.error(f"Uploaded video is {duration:.1f}s long. Please trim custom uploads to {MAX_UPLOAD_SECONDS:.0f}s or less.")
        return None

    file_hash = hashlib.sha256(in_path.read_bytes()).hexdigest()
    raw_out_mp4 = tmp_dir / "annotated_raw.mp4"
    web_out_mp4 = tmp_dir / "annotated_web.mp4"
    web_orig_mp4 = tmp_dir / "original_web.mp4"
    out_csv = tmp_dir / "detections.csv"

    use_gpu = torch.cuda.is_available()
    sample_fps = 10 if use_gpu else 3

    t0 = time.time()
    detect_video_with_model(
        input_path=in_path,
        output_path=raw_out_mp4,
        model_path=MODEL_12_PATH,
        confidence_threshold=threshold,
        analysis_fps=sample_fps,
        save_detected_frames=False,
        csv_output_path=out_csv,
        enable_cctv_intelligence=cctv_intel_enabled,
        enable_temporal_consistency=cctv_intel_enabled,
        camera_id="CUSTOM-UPLOAD",
    )
    elapsed = round(time.time() - t0, 1)

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    for src_p, dst_p in [(in_path, web_orig_mp4), (raw_out_mp4, web_out_mp4)]:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(src_p), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst_p)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    from web_app.build_web_library import parse_detections_csv
    all_recs = parse_detections_csv(out_csv, w, h)
    filtered = [
        r for r in all_recs
        if r["confidence_score"] >= threshold and (
            not cctv_intel_enabled or (r["validation_status"] in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL") and not r["rejection_reason"])
        )
    ]

    entry = {
        "video_id": f"custom_{file_hash[:8]}",
        "pair_id": f"custom_{file_hash[:8]}",
        "title": f"Custom Upload — {uploaded_file.name}",
        "version": "Original",
        "source_filename": uploaded_file.name,
        "sha256": file_hash,
        "category": "Custom Upload",
        "orig_meta": {"width": w, "height": h, "fps": round(fps, 2), "total_frames": n_frames, "duration_seconds": round(duration, 2)},
    }
    return {
        "entry": entry,
        "orig_mp4": web_orig_mp4 if web_orig_mp4.exists() else in_path,
        "annot_mp4": web_out_mp4 if web_out_mp4.exists() else raw_out_mp4,
        "records": filtered,
        "elapsed": elapsed,
        "device": "CUDA GPU" if use_gpu else "Cloud CPU",
    }


def inject_exact_desktop_css(dark_mode: bool) -> None:
    if not dark_mode:
        color_scheme = "light"
        brand_accent = "#22396f"
        focus_ring = "#22396f"
        scrollbar_thumb = "#50648b"
        scrollbar_track = "#edf2fa"
        bg_shell = "#f4f7fc"
        bg_panel = "#ffffff"
        bg_card = "#f5f8fc"
        bg_tint = "#eef3fb"
        border = "#dce4ef"
        border_tint = "#d6e2f4"
        text_main = "#0d1c42"
        text_muted = "#60718b"
        text_accent = "#22396f"
        tab_bar_bg = "#edf2fa"
        btn_bg = "#ffffff"
        btn_hover = "#eef3fb"
        btn_border = "#d7e0ed"
        empty_bg = "#0d1c42"
    else:
        color_scheme = "dark"
        brand_accent = "#9db9ee"
        focus_ring = "#9db9ee"
        scrollbar_thumb = "#8ba4cc"
        scrollbar_track = "#13223b"
        bg_shell = "#101b32"
        bg_panel = "#13223b"
        bg_card = "#1b2d49"
        bg_tint = "#1a2c50"
        border = "#354c70"
        border_tint = "#354c70"
        text_main = "#e8eef8"
        text_muted = "#aebed4"
        text_accent = "#9db9ee"
        tab_bar_bg = "#13223b"
        btn_bg = "#1b2d49"
        btn_hover = "#22396f"
        btn_border = "#354c70"
        empty_bg = "#091329"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Modern Web Guidance: Root color-scheme, brand accent-color, and accessible scrollbars */
        :root {{
            color-scheme: {color_scheme};
            accent-color: {brand_accent};
            --fk-scrollbar-thumb: {scrollbar_thumb};
            --fk-scrollbar-track: {scrollbar_track};
            scrollbar-color: var(--fk-scrollbar-thumb) var(--fk-scrollbar-track);
            scrollbar-width: thin;
        }}
        @supports not (scrollbar-color: auto) {{
            :root, .stApp, .st-key-fk_side_panel, .st-key-fk_workspace_panel, .st-key-fk_summary_panel {{
                scrollbar-width: auto;
            }}
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            ::-webkit-scrollbar-thumb {{
                background: var(--fk-scrollbar-thumb);
                border-radius: 4px;
            }}
            ::-webkit-scrollbar-track {{
                background: var(--fk-scrollbar-track);
            }}
        }}

        /* Modern Web Guidance: Natural interactions & accessible :focus-visible rings */
        :where(a, button, input, select, textarea, summary, [tabindex]):focus-visible {{
            outline: 2px solid {focus_ring} !important;
            outline-offset: 2px !important;
        }}
        .st-key-fk_header_bar :where(a, button, [tabindex]):focus-visible {{
            outline-color: #9db9ee !important;
        }}
        @media (prefers-reduced-motion: no-preference) {{
            div.stButton > button, div.stDownloadButton > button {{
                transition: background-color 150ms ease-out, border-color 150ms ease-out, box-shadow 150ms ease-out, scale 150ms ease-out;
            }}
            div.stButton > button:not(:disabled):active, div.stDownloadButton > button:not(:disabled):active {{
                scale: 0.98;
            }}
        }}

        /* Remove Streamlit chrome and default spacing */
        header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
            display: none !important;
        }}
        .stApp {{
            background-color: {bg_shell};
            font-family: "Inter", "Segoe UI", sans-serif;
            color: {text_main};
        }}
        .block-container {{
            padding: 0 !important;
            max-width: 100% !important;
        }}

        /* ===== EXACT DESKTOP TOP HEADER BAR (#header) ===== */
        .st-key-fk_header_bar {{
            color-scheme: dark;
            background: #0d1c42 !important;
            border-bottom: 1px solid #22396f !important;
            padding: 9px 20px !important;
            margin: 0 !important;
        }}
        .st-key-fk_header_bar [data-testid="stHorizontalBlock"] {{
            align-items: center !important;
            gap: 8px !important;
        }}
        .fk-brand-wrap {{
            display: flex;
            align-items: center;
            gap: 11px;
        }}
        .fk-brand-logo {{
            height: 38px;
            width: auto;
            object-fit: contain;
        }}
        .fk-brand-title {{
            color: #ffffff;
            font-size: 19px;
            font-weight: 700;
            letter-spacing: 0.7px;
            line-height: 1.1;
        }}
        .fk-brand-sub {{
            color: #c7cdd5;
            font-size: 9px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin-top: 2px;
        }}
        /* Header buttons match QPushButton#headerButton and #primaryButton */
        .st-key-fk_header_bar div.stButton > button,
        .st-key-fk_header_bar div.stDownloadButton > button {{
            background-color: #172950 !important;
            color: #ffffff !important;
            border: 1px solid #50648b !important;
            border-radius: 10px !important;
            font-size: 12.5px !important;
            font-weight: 500 !important;
            min-height: 38px !important;
            padding: 4px 12px !important;
            white-space: nowrap !important;
        }}
        .st-key-fk_header_bar div.stButton > button:hover,
        .st-key-fk_header_bar div.stDownloadButton > button:hover {{
            background-color: #22396f !important;
            border-color: #9db9ee !important;
            color: #ffffff !important;
        }}
        .st-key-fk_header_bar div.stButton > button[kind="primary"] {{
            background-color: #22396f !important;
            border: 1px solid #506b9e !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }}
        .st-key-fk_header_bar div.stButton > button[kind="primary"]:hover {{
            background-color: #304d85 !important;
            border-color: #9db9ee !important;
        }}
        .st-key-fk_header_bar div.stButton > button:disabled,
        .st-key-fk_header_bar div.stDownloadButton > button:disabled {{
            background-color: #0d1c42 !important;
            border-color: #32486e !important;
            color: #7c8da8 !important;
            opacity: 1 !important;
        }}

        /* ===== EXACT DESKTOP MODE TAB STRIP (#modeTabs) ===== */
        .st-key-fk_mode_bar {{
            background: {tab_bar_bg} !important;
            border-bottom: 1px solid {border} !important;
            padding: 6px 20px !important;
            margin: 0 !important;
        }}
        .st-key-fk_mode_bar [data-testid="stHorizontalBlock"] {{
            gap: 6px !important;
            align-items: center !important;
        }}
        .st-key-fk_mode_bar div.stButton > button {{
            background: transparent !important;
            border: none !important;
            color: {text_muted} !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            border-radius: 9px !important;
            padding: 6px 16px !important;
            min-height: 34px !important;
        }}
        .st-key-fk_mode_bar div.stButton > button:hover {{
            color: {text_accent} !important;
            background: rgba(34, 57, 111, 0.08) !important;
        }}
        .st-key-fk_mode_bar div.stButton > button[kind="primary"] {{
            background: #22396f !important;
            color: #ffffff !important;
            border-radius: 9px !important;
        }}

        /* ===== EXACT 3-COLUMN FLUSH DESKTOP BODY ===== */
        .st-key-fk_body_wrap {{
            padding: 0 !important;
            margin: 0 !important;
        }}
        .st-key-fk_side_panel {{
            background: {bg_panel} !important;
            border-right: 1px solid {border} !important;
            padding: 14px 15px 18px 16px !important;
            min-height: calc(100vh - 108px) !important;
        }}
        .st-key-fk_workspace_panel {{
            background: {bg_shell} !important;
            padding: 14px 16px 14px 16px !important;
            min-height: calc(100vh - 108px) !important;
        }}
        .st-key-fk_summary_panel {{
            background: {bg_panel} !important;
            border-left: 1px solid {border} !important;
            padding: 14px 16px 18px 16px !important;
            min-height: calc(100vh - 108px) !important;
        }}

        /* ===== DESKTOP TYPOGRAPHY & CARDS ===== */
        .fk-sec {{
            color: {text_main};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.45px;
            text-transform: uppercase;
            margin: 12px 0 6px 0;
        }}
        .fk-sec-first {{
            margin-top: 0 !important;
        }}
        .fk-evidence-card {{
            background: {bg_tint};
            border: 1px solid {border_tint};
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }}
        .fk-evidence-title {{
            color: {text_accent};
            font-size: 13px;
            font-weight: 600;
            word-break: break-word;
            line-height: 1.3;
        }}
        .fk-muted {{
            color: {text_muted};
            font-size: 11px;
            line-height: 1.4;
        }}
        .fk-enh-card {{
            background: {bg_card};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 4px;
        }}
        .fk-enh-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }}
        .fk-enh-title {{
            color: {text_main};
            font-size: 13px;
            font-weight: 600;
        }}
        .fk-enh-badge {{
            background: {bg_tint};
            color: {text_accent};
            border-radius: 6px;
            padding: 2px 7px;
            font-size: 11px;
            font-weight: 600;
        }}
        .fk-status-card {{
            background: {bg_card};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 10px 12px;
            margin-top: 2px;
        }}
        .fk-status-title {{
            color: {text_main};
            font-size: 12px;
            font-weight: 650;
        }}

        /* Segmented View Mode Toggle (#toggleBox) */
        .st-key-fk_toggle_box, .st-key-fk_mc_toggle_box {{
            background: {tab_bar_bg} !important;
            border: 1px solid {border} !important;
            border-radius: 11px !important;
            padding: 4px !important;
        }}
        .st-key-fk_toggle_box div.stButton > button,
        .st-key-fk_mc_toggle_box div.stButton > button {{
            background: transparent !important;
            border: 1px solid transparent !important;
            color: {text_muted} !important;
            border-radius: 8px !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            min-height: 32px !important;
            padding: 4px 6px !important;
        }}
        .st-key-fk_toggle_box div.stButton > button[kind="primary"],
        .st-key-fk_mc_toggle_box div.stButton > button[kind="primary"] {{
            background: {bg_panel} !important;
            border: 1px solid {border_tint} !important;
            color: {text_accent} !important;
            font-weight: 600 !important;
        }}

        /* Workspace heading, empty cinema canvas, observations panel */
        .fk-ws-heading {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .fk-ws-title {{
            color: {text_main};
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }}
        .fk-empty-canvas {{
            background: {empty_bg};
            border: 1px solid #22396f;
            border-radius: 6px;
            min-height: 410px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #aebed4;
            gap: 10px;
            padding: 32px;
            margin-bottom: 12px;
        }}
        .fk-empty-canvas .fk-ec-main {{
            font-size: 14px;
            color: #d6e2f4;
            font-weight: 500;
        }}
        .fk-empty-canvas .fk-ec-sub {{
            font-size: 12.5px;
            color: #8f9fbc;
        }}
        .st-key-fk_obs_box {{
            background: {bg_panel} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
            padding: 10px 12px 8px 12px !important;
            margin-top: 8px !important;
        }}
        .fk-safety-note {{
            color: #7d735b;
            font-size: 10.5px;
            font-weight: 500;
        }}

        /* Summary panel metric cards (#metricCard) */
        .fk-metrics-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 7px;
            margin-bottom: 10px;
        }}
        .fk-metric-card {{
            background: {bg_card};
            border: 1px solid {border};
            border-radius: 11px;
            padding: 10px 6px;
            text-align: center;
        }}
        .fk-metric-num {{
            color: {text_main};
            font-size: 26px;
            font-weight: 650;
            line-height: 1.1;
        }}
        .fk-metric-lbl {{
            color: {text_muted};
            font-size: 9px;
            font-weight: 600;
            letter-spacing: 0.4px;
            margin-top: 3px;
        }}
        .fk-det-card {{
            background: {bg_panel};
            border: 1px solid {border};
            border-radius: 11px;
            padding: 9px 11px;
            margin-bottom: 8px;
        }}
        .fk-det-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .fk-det-name {{
            color: {text_main};
            font-size: 13px;
            font-weight: 600;
        }}
        .fk-conf-pill {{
            background: #e5eddf;
            color: #427432;
            border-radius: 6px;
            padding: 2px 7px;
            font-size: 11px;
            font-weight: 700;
        }}
        .fk-dec-accept {{ background: #256044; color: #fff; border-radius: 6px; padding: 2px 7px; font-size: 10.5px; font-weight: 600; }}
        .fk-dec-reject {{ background: #9d343a; color: #fff; border-radius: 6px; padding: 2px 7px; font-size: 10.5px; font-weight: 600; }}
        .fk-dec-uncertain {{ background: #865710; color: #fff; border-radius: 6px; padding: 2px 7px; font-size: 10.5px; font-weight: 600; }}
        .fk-dec-pending {{ background: #fff2d8; color: #775014; border: 1px solid #efd8aa; border-radius: 6px; padding: 2px 7px; font-size: 10.5px; font-weight: 600; }}

        /* Standard desktop buttons inside body */
        div.stButton > button, div.stDownloadButton > button {{
            border-radius: 11px !important;
            font-weight: 500 !important;
            font-size: 12.5px !important;
            background-color: {btn_bg} !important;
            color: {text_main} !important;
            border: 1px solid {btn_border} !important;
            min-height: 36px !important;
        }}
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            background-color: {btn_hover} !important;
            border-color: #91a9d0 !important;
            color: {text_main} !important;
        }}
        div.stButton > button p, div.stDownloadButton > button p {{
            color: inherit !important;
        }}
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {{
            background-color: #22396f !important;
            border-color: #22396f !important;
            color: #ffffff !important;
        }}
        div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button[kind="primary"]:hover {{
            background-color: #304d85 !important;
            color: #ffffff !important;
        }}
        div.stButton > button:disabled, div.stDownloadButton > button:disabled {{
            background-color: #eef3fb !important;
            color: #8190a6 !important;
            border-color: #dae3ef !important;
            opacity: 1 !important;
        }}

        /* Clean light selectboxes, inputs, video player */
        label, .stRadio label, .stRadio p, .stToggle label, .stToggle p, .stSelectbox label {{
            color: {text_main} !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: {bg_panel} !important;
            border: 1px solid {btn_border} !important;
            border-radius: 10px !important;
            color: {text_main} !important;
        }}
        div[data-baseweb="select"] span, div[data-baseweb="select"] svg {{
            color: {text_main} !important;
            fill: {text_main} !important;
        }}
        div[data-testid="stVideo"] {{
            color-scheme: dark;
            background: #0d1c42;
            border: 1px solid #22396f;
            border-radius: 6px;
            overflow: hidden;
        }}
        code {{
            background: transparent !important;
            color: {text_muted} !important;
            padding: 0 !important;
            font-size: 11px !important;
        }}

        /* ===== BRANDED SPLASH / LOADING OVERLAY ON OPEN & REFRESH ===== */
        .fk-splash-overlay {{
            position: fixed;
            inset: 0;
            z-index: 99999;
            background: #0d1c42;
            color-scheme: dark;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 14px;
            pointer-events: all;
            animation: fk-splash-fade 0.45s ease-out 1.15s forwards;
        }}
        .fk-splash-logo {{
            height: 76px;
            width: auto;
            object-fit: contain;
            animation: fk-splash-pop 0.45s ease-out both;
        }}
        .fk-splash-title {{
            color: #ffffff;
            font-size: 26px;
            font-weight: 700;
            letter-spacing: 1.2px;
            line-height: 1.1;
        }}
        .fk-splash-sub {{
            color: #9db9ee;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.4px;
            text-transform: uppercase;
        }}
        .fk-splash-track {{
            width: 210px;
            height: 4px;
            background: rgba(157, 185, 238, 0.22);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 6px;
        }}
        .fk-splash-bar {{
            width: 100%;
            height: 100%;
            background: #9db9ee;
            border-radius: 999px;
            transform-origin: left center;
            animation: fk-splash-load 1.05s ease-in-out forwards;
        }}
        @keyframes fk-splash-pop {{
            from {{ opacity: 0; transform: scale(0.92); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        @keyframes fk-splash-load {{
            from {{ transform: scaleX(0.08); }}
            to {{ transform: scaleX(1); }}
        }}
        @keyframes fk-splash-fade {{
            from {{ opacity: 1; visibility: visible; }}
            to {{ opacity: 0; visibility: hidden; pointer-events: none; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .fk-splash-overlay {{ animation-duration: 0.15s; animation-delay: 0.4s; }}
            .fk-splash-logo, .fk-splash-bar {{ animation: none; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _dec_badge(decision: str) -> str:
    cls = {
        "Accept": "fk-dec-accept",
        "Reject": "fk-dec-reject",
        "Uncertain": "fk-dec-uncertain",
    }.get(decision, "fk-dec-pending")
    return f"<span class='{cls}'>{decision}</span>"


def get_active_single_context(single_videos: list[dict]):
    ss = st.session_state
    threshold = ss["conf_pct"] / 100.0
    cctv_intel = ss["cctv_intel"]
    analysis_ready = bool(ss.get("analysis_ready", False))

    if ss.get("custom_result") is not None:
        cr = ss["custom_result"]
        return cr["entry"], cr["orig_mp4"], cr["annot_mp4"], (cr["records"] if analysis_ready else []), None

    vid_id = ss.get("active_video_id")
    if not vid_id:
        return None, None, None, [], None

    entry = next((v for v in single_videos if v["video_id"] == vid_id), None)
    if entry is None:
        return None, None, None, [], None

    orig_mp4 = WEB_DATA_DIR / entry["original_video_rel"]
    annot_mp4 = WEB_DATA_DIR / entry["annotated_video_rel"]
    if analysis_ready:
        telem = load_telemetry(entry["telemetry_rel"])
        recs = filter_records(telem, threshold, cctv_intel)
    else:
        recs = []

    # Check if there is a paired Original <-> Enhanced version in the library
    pair_id = entry.get("pair_id")
    cur_ver = entry.get("version", "Original")
    alt_ver = "Enhanced" if cur_ver == "Original" else "Original"
    paired_entry = next(
        (v for v in single_videos if v.get("pair_id") == pair_id and v.get("version") == alt_ver),
        None,
    )
    return entry, orig_mp4, annot_mp4, recs, paired_entry


# ==================== DESKTOP-MATCHED MODAL DIALOGS ====================

@st.dialog("Import CCTV Videos", width="large")
def dialog_import_videos(single_videos: list[dict], mc_scenes: list[dict]) -> None:
    st.markdown(
        "<div class='fk-muted' style='margin-bottom:10px;'>Choose a pre-indexed CCTV recording from the Thesis Evaluation Library for instant playback, or upload your own short CCTV clip (≤25 MB, ≤15s) for live Model 12 (Video SOTA) inference.</div>",
        unsafe_allow_html=True,
    )
    t_lib, t_mc, t_up = st.tabs(["Prepared CCTV Library (15 videos)", "Multi-Camera Pairs (6 scenes)", "Upload Custom Clip (≤25 MB)"])

    with t_lib:
        cats = ["All"] + sorted(list({v["category"] for v in single_videos}))
        sel_cat = st.selectbox("Filter Category", cats, key="dlg_cat")
        vids = single_videos if sel_cat == "All" else [v for v in single_videos if v["category"] == sel_cat]
        if vids:
            sel_i = st.selectbox(
                "Select Recording",
                range(len(vids)),
                format_func=lambda i: vids[i]["title"],
                key="dlg_vid_idx",
            )
            chosen = vids[sel_i]
            om = chosen["orig_meta"]
            st.caption(f"File: `{chosen['source_filename']}` · {om['width']}×{om['height']} · {om['fps']} fps · {om['duration_seconds']}s · SHA-256: `{chosen['sha256'][:16]}…`")
            if st.button("Load Selected Recording", type="primary", width="stretch", key="dlg_load_lib"):
                st.session_state["custom_result"] = None
                st.session_state["active_video_id"] = chosen["video_id"]
                st.session_state["active_mode"] = "Single video"
                st.session_state["view_mode"] = "Original"
                st.session_state["analysis_ready"] = False
                st.session_state["selected_obs_idx"] = 0
                st.rerun()

    with t_mc:
        if mc_scenes:
            sel_m = st.selectbox(
                "Select Synchronized CAM-01 + CAM-02 Scene",
                range(len(mc_scenes)),
                format_func=lambda i: mc_scenes[i]["title"],
                key="dlg_mc_idx",
            )
            if st.button("Open in Multi-Camera Workspace", type="primary", width="stretch", key="dlg_load_mc"):
                st.session_state["active_mc_key"] = mc_scenes[sel_m]["scene_key"]
                st.session_state["active_mode"] = "Multi-camera"
                st.rerun()

    with t_up:
        up = st.file_uploader(
            "Select CCTV footage (MP4, AVI, MOV, MKV, WebM — max 25 MB / 15s)",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="dlg_uploader",
        )
        if st.button("Run Live Model 12 Analysis", type="primary", width="stretch", disabled=(up is None), key="dlg_run_up"):
            with st.spinner("Running Model 12 + CCTV Intelligence Layer on uploaded clip..."):
                res = run_live_custom_upload(up, st.session_state["conf_pct"] / 100.0, st.session_state["cctv_intel"])
                if res:
                    st.session_state["custom_result"] = res
                    st.session_state["active_video_id"] = res["entry"]["video_id"]
                    st.session_state["active_mode"] = "Single video"
                    st.session_state["view_mode"] = "Detected"
                    st.session_state["analysis_ready"] = True
                    st.session_state["selected_obs_idx"] = 0
                    st.rerun()


@st.dialog("Observation Review", width="large")
def dialog_observation_review(entry: dict, orig_mp4: Path, filtered_records: list[dict]) -> None:
    if not filtered_records:
        st.info("No weapon detections were flagged at the current threshold and filter configuration.")
        return

    vid_id = entry["video_id"]
    vid_reviews = st.session_state["reviews"].setdefault(vid_id, {})

    options = [
        f"#{i+1:02d} · {r['timestamp_formatted']} · Frame {r['frame_number']} · {r['object_label'].title()} ({r['confidence_score']*100:.1f}%)"
        for i, r in enumerate(filtered_records)
    ]
    default_i = min(st.session_state.get("selected_obs_idx", 0), len(options) - 1)
    sel_i = st.selectbox("Select Observation to Review", range(len(options)), index=default_i, format_func=lambda i: options[i], key="rev_dlg_pick")
    st.session_state["selected_obs_idx"] = sel_i

    rec = filtered_records[sel_i]
    rkey = f"{rec['frame_number']}_{rec['object_label']}_{sel_i}"
    cur_rev = vid_reviews.get(rkey, {})
    cur_dec = cur_rev.get("decision", "Pending")

    c_img, c_ctrl = st.columns([1.3, 1.0], gap="medium")
    with c_img:
        if orig_mp4 and orig_mp4.exists():
            frame_rgb = extract_annotated_frame_rgb(orig_mp4, rec["frame_number"], rec["norm_box"], rec["object_label"], rec["confidence_score"])
            if frame_rgb is not None:
                st.image(frame_rgb, caption=f"Frame {rec['frame_number']} ({rec['timestamp_formatted']}) · Box {rec['bounding_box']}", width="stretch")

    with c_ctrl:
        st.markdown(
            f"""
            <div class='fk-det-card'>
                <div class='fk-det-top'>
                    <span class='fk-det-name'>{rec['object_label'].title()}</span>
                    <span class='fk-conf-pill'>{rec['confidence_score']*100:.1f}%</span>
                </div>
                <div class='fk-muted' style='margin-top:5px;'>
                    Timestamp: <b>{rec['timestamp_formatted']}</b> · Frame <b>{rec['frame_number']}</b><br/>
                    Status: {_dec_badge(cur_dec)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        notes = st.text_area("Analyst Notes (optional)", value=cur_rev.get("notes", ""), height=90, key=f"rev_notes_{vid_id}_{sel_i}")
        b1, b2, b3 = st.columns(3)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if b1.button("Accept", key=f"dlg_acc_{vid_id}_{sel_i}", width="stretch"):
            vid_reviews[rkey] = {"decision": "Accept", "notes": notes, "reviewedAt": now_utc}
            st.rerun()
        if b2.button("Reject", key=f"dlg_rej_{vid_id}_{sel_i}", width="stretch"):
            vid_reviews[rkey] = {"decision": "Reject", "notes": notes, "reviewedAt": now_utc}
            st.rerun()
        if b3.button("Uncertain", key=f"dlg_unc_{vid_id}_{sel_i}", width="stretch"):
            vid_reviews[rkey] = {"decision": "Uncertain", "notes": notes, "reviewedAt": now_utc}
            st.rerun()


@st.dialog("Forensic Report", width="large")
def dialog_forensic_report(entry: dict, filtered_records: list[dict], threshold: float, cctv_intel: bool) -> None:
    vid_id = entry["video_id"]
    vid_reviews = st.session_state["reviews"].setdefault(vid_id, {})
    om = entry.get("orig_meta", {})

    st.markdown(f"**Evidence File:** `{entry.get('source_filename', '')}`")
    st.markdown(f"**SHA-256 Chain-of-Custody Hash:** `{entry.get('sha256', '')}`")
    st.caption(f"Resolution: {om.get('width', 0)}×{om.get('height', 0)} · {om.get('fps', 0)} fps · {om.get('duration_seconds', 0)}s | Model 12 (Video SOTA) | Threshold: {int(threshold*100)}% | CCTV Intelligence: {'ON' if cctv_intel else 'OFF'}")

    hg_cnt = sum(1 for r in filtered_records if r["object_label"] == "handgun")
    kn_cnt = sum(1 for r in filtered_records if r["object_label"] == "knife")
    st.markdown(
        f"""
        <div class='fk-metrics-row'>
            <div class='fk-metric-card'><div class='fk-metric-num'>{len(filtered_records)}</div><div class='fk-metric-lbl'>DETECTIONS</div></div>
            <div class='fk-metric-card'><div class='fk-metric-num'>{hg_cnt}</div><div class='fk-metric-lbl'>HANDGUNS</div></div>
            <div class='fk-metric-card'><div class='fk-metric-num'>{kn_cnt}</div><div class='fk-metric-lbl'>KNIVES</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_records:
        df_rep = pd.DataFrame([
            {
                "Time": r["timestamp_formatted"],
                "Frame": r["frame_number"],
                "Object": r["object_label"].title(),
                "Confidence": f"{r['confidence_score']*100:.1f}%",
                "Bounding box": str(r["bounding_box"]),
                "Analyst decision": vid_reviews.get(f"{r['frame_number']}_{r['object_label']}_{i}", {}).get("decision", "Not reviewed"),
                "Analyst notes": vid_reviews.get(f"{r['frame_number']}_{r['object_label']}_{i}", {}).get("notes", ""),
            }
            for i, r in enumerate(filtered_records)
        ])
        st.dataframe(df_rep, width="stretch", hide_index=True, height=240)
    else:
        st.info("No confirmed weapon detections in this recording under active filter settings.")

    pdf_bytes = build_single_video_pdf_bytes(entry, filtered_records, threshold, cctv_intel, vid_reviews)
    csv_bytes = (
        pd.DataFrame(filtered_records).to_csv(index=False).encode("utf-8")
        if filtered_records
        else b"frame_number,timestamp_formatted,object_label,confidence_score,bounding_box\n"
    )
    d1, d2 = st.columns(2)
    d1.download_button("Download Forensic PDF Report", data=pdf_bytes, file_name=f"FORENSIKADA_{vid_id}_report.pdf", mime="application/pdf", type="primary", width="stretch")
    d2.download_button("Download Detection Log (CSV)", data=csv_bytes, file_name=f"FORENSIKADA_{vid_id}_detections.csv", mime="text/csv", width="stretch")


@st.dialog("System Settings")
def dialog_settings() -> None:
    st.markdown("<div class='fk-sec fk-sec-first'>APPEARANCE</div>", unsafe_allow_html=True)
    theme_choice = st.radio(
        "Theme",
        ["Light (Default)", "Dark"],
        index=0 if st.session_state["theme"] == "light" else 1,
        horizontal=True,
        key="set_theme_radio",
    )
    st.markdown("<div class='fk-sec'>CCTV INTELLIGENCE LAYER</div>", unsafe_allow_html=True)
    intel_val = st.toggle(
        "Enable Person-Proximity Gate + Temporal Consistency (min_hits=3)",
        value=st.session_state["cctv_intel"],
        key="set_intel_toggle",
    )
    if st.button("Save Settings", type="primary", width="stretch"):
        st.session_state["theme"] = "light" if theme_choice.startswith("Light") else "dark"
        st.session_state["cctv_intel"] = intel_val
        st.rerun()


# ==================== MAIN DESKTOP PAGES ====================

def render_single_video_page(single_videos: list[dict], mc_scenes: list[dict], entry, orig_mp4, annot_mp4, filtered_records, paired_entry) -> None:
    ss = st.session_state
    threshold = ss["conf_pct"] / 100.0

    with st.container(key="fk_body_wrap"):
        col_side, col_work, col_sum = st.columns([0.92, 2.30, 1.02], gap="small")

        # ---------- LEFT: sidePanel ----------
        with col_side:
            with st.container(key="fk_side_panel"):
                st.markdown("<div class='fk-sec fk-sec-first'>EVIDENCE SOURCE</div>", unsafe_allow_html=True)
                if entry is not None:
                    om = entry.get("orig_meta", {})
                    st.markdown(
                        f"""
                        <div class='fk-evidence-card'>
                            <div class='fk-evidence-title'>{entry.get('source_filename', '')}</div>
                            <div class='fk-muted'>{om.get('width', 0)}×{om.get('height', 0)} · {om.get('fps', 0)} fps · {om.get('duration_seconds', 0)}s</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class='fk-evidence-card'>
                            <div class='fk-evidence-title'>No video selected</div>
                            <div class='fk-muted'>Import CCTV footage</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Quick Library Dropdown right below Evidence Source for 1-click evaluator switching
                analysis_ready = bool(ss.get("analysis_ready", False))
                if single_videos:
                    lib_ids = [v["video_id"] for v in single_videos]
                    options_idx = [-1] + list(range(len(single_videos)))
                    cur_sel = (lib_ids.index(entry["video_id"]) + 1) if (entry and entry["video_id"] in lib_ids) else 0
                    picked_opt = st.selectbox(
                        "Quick Switch Library Clip",
                        options_idx,
                        index=cur_sel,
                        format_func=lambda idx: "— Choose a CCTV recording —" if idx == -1 else single_videos[idx]["title"],
                        label_visibility="collapsed",
                        key="side_quick_pick",
                    )
                    if picked_opt != -1 and (entry is None or single_videos[picked_opt]["video_id"] != entry["video_id"]) and ss.get("custom_result") is None:
                        ss["active_video_id"] = single_videos[picked_opt]["video_id"]
                        ss["view_mode"] = "Original"
                        ss["analysis_ready"] = False
                        ss["selected_obs_idx"] = 0
                        st.rerun()
                    elif picked_opt == -1 and entry is not None and ss.get("custom_result") is None:
                        ss["active_video_id"] = None
                        ss["view_mode"] = "Original"
                        ss["analysis_ready"] = False
                        ss["selected_obs_idx"] = 0
                        st.rerun()

                # Enhancement (BasicVSR++) card
                is_enh = bool(entry and entry.get("version") == "Enhanced")
                enh_sub = (
                    "BasicVSR++ super-resolution active."
                    if is_enh
                    else ("BasicVSR++ enhanced pair available." if paired_entry else "Import a video to enhance it.")
                )
                st.markdown(
                    f"""
                    <div class='fk-enh-card'>
                        <div class='fk-enh-top'>
                            <span class='fk-enh-title'>Enhancement</span>
                            <span class='fk-enh-badge'>BasicVSR++</span>
                        </div>
                        <div class='fk-muted'>{enh_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                enh_btn_label = "Switch to Original" if (is_enh and paired_entry) else "Enhance video"
                if st.button(enh_btn_label, width="stretch", disabled=(paired_entry is None), key="btn_enhance_pair"):
                    ss["active_video_id"] = paired_entry["video_id"]
                    ss["selected_obs_idx"] = 0
                    st.rerun()

                # VIEW MODE segmented pill (#toggleBox)
                st.markdown("<div class='fk-sec'>VIEW MODE</div>", unsafe_allow_html=True)
                with st.container(key="fk_toggle_box"):
                    vt1, vt2 = st.columns(2, gap="small")
                    if vt1.button("Original", type=("primary" if ss["view_mode"] == "Original" else "secondary"), width="stretch", disabled=(entry is None), key="vm_orig"):
                        ss["view_mode"] = "Original"
                        st.rerun()
                    if vt2.button("Detected", type=("primary" if (ss["view_mode"] == "Detected" and analysis_ready) else "secondary"), width="stretch", disabled=(entry is None or not analysis_ready), key="vm_det"):
                        ss["view_mode"] = "Detected"
                        st.rerun()

                # CONFIDENCE THRESHOLD
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; margin:12px 0 2px 0;'>
                        <span class='fk-sec' style='margin:0;'>CONFIDENCE THRESHOLD</span>
                        <span style='color:#22396f; font-size:12.5px; font-weight:700;'>{ss['conf_pct']}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                new_conf = st.slider(
                    "Confidence",
                    min_value=20,
                    max_value=90,
                    value=ss["conf_pct"],
                    step=5,
                    label_visibility="collapsed",
                    key="side_conf_slider",
                )
                if new_conf != ss["conf_pct"]:
                    ss["conf_pct"] = new_conf
                    ss["selected_obs_idx"] = 0
                    st.rerun()

                # Desktop action stack under slider
                analyze_btn_type = "primary" if (entry is not None and not analysis_ready) else "secondary"
                if st.button("Analyze video", type=analyze_btn_type, width="stretch", disabled=(entry is None), key="btn_analyze_side"):
                    with st.spinner(f"Analyzing video at {ss['conf_pct']}% threshold..."):
                        time.sleep(0.65)
                    ss["analysis_ready"] = True
                    ss["view_mode"] = "Detected"
                    ss["selected_obs_idx"] = 0
                    st.rerun()
                if st.button("Observation review", width="stretch", disabled=(entry is None or not analysis_ready), key="btn_obs_rev_side"):
                    dialog_observation_review(entry, orig_mp4, filtered_records)

                new_intel = st.toggle("CCTV Intelligence Layer", value=ss["cctv_intel"], key="side_intel_toggle")
                if new_intel != ss["cctv_intel"]:
                    ss["cctv_intel"] = new_intel
                    ss["selected_obs_idx"] = 0
                    st.rerun()

                # ANALYSIS STATUS
                st.markdown("<div class='fk-sec'>ANALYSIS STATUS</div>", unsafe_allow_html=True)
                if entry is None:
                    st_title = "Ready for detection"
                    st_detail = "Choose or import a CCTV video to begin."
                elif not analysis_ready:
                    st_title = "Video loaded"
                    st_detail = f"Set threshold ({ss['conf_pct']}%), then click Analyze video."
                else:
                    st_title = "Analysis complete"
                    st_detail = f"{len(filtered_records)} observation(s) at {ss['conf_pct']}% · {'CCTV Intel ON' if ss['cctv_intel'] else 'Raw Detector'}"
                st.markdown(
                    f"""
                    <div class='fk-status-card'>
                        <div class='fk-status-title'>{st_title}</div>
                        <div class='fk-muted'>{st_detail}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # TRAINED MODEL footer
                st.markdown("<div class='fk-sec' style='margin-top:20px;'>TRAINED MODEL</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='fk-muted'>Model 12 (Production Video SOTA)<br/>Faster R-CNN · Handgun / Knife<br/>best_weapon_detector_twelfth_model.pth</div>",
                    unsafe_allow_html=True,
                )

        # ---------- CENTER: workspace ----------
        with col_work:
            with st.container(key="fk_workspace_panel"):
                ws_meta = (
                    "Original evidence"
                    if (entry is None or ss["view_mode"] == "Original")
                    else "Detected evidence · Model 12"
                )
                st.markdown(
                    f"""
                    <div class='fk-ws-heading'>
                        <span class='fk-ws-title'>VIDEO WEAPON DETECTION</span>
                        <span class='fk-muted'>{ws_meta}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if entry is None:
                    st.markdown(
                        """
                        <div class='fk-empty-canvas'>
                            <div class='fk-ec-main'>Import a CCTV video to begin</div>
                            <div class='fk-ec-sub'>MP4, AVI, MOV, MKV or WebM</div>
                            <div class='fk-ec-sub'>Enhance if needed, then analyze</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    play_path = annot_mp4 if ss["view_mode"] == "Detected" else orig_mp4
                    if play_path is not None and play_path.exists():
                        st.video(str(play_path))
                    else:
                        st.warning("Video stream unavailable.")

                # DETECTED OBJECTS · CLICK A ROW TO REVIEW THE FRAME
                with st.container(key="fk_obs_box"):
                    st.markdown("<div class='fk-sec fk-sec-first'>DETECTED OBJECTS · CLICK A ROW TO REVIEW THE FRAME</div>", unsafe_allow_html=True)
                    if filtered_records:
                        df_obs = pd.DataFrame([
                            {
                                "Time": r["timestamp_formatted"],
                                "Object": r["object_label"].title(),
                                "Confidence": f"{r['confidence_score']*100:.1f}%",
                                "Frame": r["frame_number"],
                            }
                            for r in filtered_records
                        ])
                        ev = st.dataframe(
                            df_obs,
                            width="stretch",
                            hide_index=True,
                            height=185,
                            on_select="rerun",
                            selection_mode="single-row",
                            key=f"obs_tbl_{entry['video_id']}_{ss['conf_pct']}_{ss['cctv_intel']}",
                        )
                        if ev and ev.selection and ev.selection.rows:
                            ss["selected_obs_idx"] = ev.selection.rows[0]
                    else:
                        empty_df = pd.DataFrame(columns=["Time", "Object", "Confidence", "Frame"])
                        st.dataframe(empty_df, width="stretch", hide_index=True, height=160)

                dim_txt = (
                    f"{entry['orig_meta']['width']}×{entry['orig_meta']['height']} · {entry['orig_meta']['fps']} fps · SHA-256: {entry['sha256'][:14]}…"
                    if entry is not None
                    else "—"
                )
                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-top:8px;'>
                        <span class='fk-safety-note'>Automated detections require human review.</span>
                        <span class='fk-muted'>{dim_txt}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---------- RIGHT: summaryPanel ----------
        with col_sum:
            with st.container(key="fk_summary_panel"):
                st.markdown("<div class='fk-sec fk-sec-first'>DETECTION SUMMARY</div>", unsafe_allow_html=True)
                has_results = (entry is not None) and bool(ss.get("analysis_ready", False))
                hg_cnt = sum(1 for r in filtered_records if r["object_label"] == "handgun")
                kn_cnt = sum(1 for r in filtered_records if r["object_label"] == "knife")
                tot_txt = str(len(filtered_records)) if has_results else "—"
                hg_txt = str(hg_cnt) if has_results else "—"
                kn_txt = str(kn_cnt) if has_results else "—"

                st.markdown(
                    f"""
                    <div class='fk-metrics-row'>
                        <div class='fk-metric-card'><div class='fk-metric-num'>{tot_txt}</div><div class='fk-metric-lbl'>DETECTIONS</div></div>
                        <div class='fk-metric-card'><div class='fk-metric-num'>{hg_txt}</div><div class='fk-metric-lbl'>HANDGUNS</div></div>
                        <div class='fk-metric-card'><div class='fk-metric-num'>{kn_txt}</div><div class='fk-metric-lbl'>KNIVES</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if entry is None:
                    st.markdown("<div class='fk-muted' style='margin-bottom:14px;'>Import or select a video, choose a confidence threshold, then click Analyze video.</div>", unsafe_allow_html=True)
                    st.markdown("<div class='fk-sec'>CURRENT FRAME</div>", unsafe_allow_html=True)
                    st.markdown("<div class='fk-muted'>No video selected yet.</div>", unsafe_allow_html=True)
                elif not has_results:
                    st.markdown(f"<div class='fk-muted' style='margin-bottom:14px;'>Video loaded. Adjust the confidence threshold (<b>{ss['conf_pct']}%</b>) and click <b>Analyze video</b> to load detections.</div>", unsafe_allow_html=True)
                    st.markdown("<div class='fk-sec'>CURRENT FRAME</div>", unsafe_allow_html=True)
                    st.markdown("<div class='fk-muted'>No detections loaded yet.</div>", unsafe_allow_html=True)
                else:
                    vid_id = entry["video_id"]
                    vid_reviews = ss["reviews"].setdefault(vid_id, {})
                    if not filtered_records:
                        st.markdown("<div class='fk-muted' style='margin-bottom:12px;'>No weapon threats flagged at the current threshold and filter configuration.</div>", unsafe_allow_html=True)
                        st.markdown("<div class='fk-sec'>CURRENT FRAME</div>", unsafe_allow_html=True)
                        st.markdown("<div class='fk-muted'>Zero detections to inspect.</div>", unsafe_allow_html=True)
                    else:
                        mean_c = sum(r["confidence_score"] for r in filtered_records) / len(filtered_records) * 100.0
                        st.markdown(f"<div class='fk-muted' style='margin-bottom:10px;'>{len(filtered_records)} frame observations (mean {mean_c:.1f}%).</div>", unsafe_allow_html=True)

                        st.markdown("<div class='fk-sec'>CURRENT FRAME</div>", unsafe_allow_html=True)
                        obs_i = min(ss.get("selected_obs_idx", 0), len(filtered_records) - 1)
                        rec = filtered_records[obs_i]
                        rkey = f"{rec['frame_number']}_{rec['object_label']}_{obs_i}"
                        cur_dec = vid_reviews.get(rkey, {}).get("decision", "Pending")

                        st.markdown(
                            f"""
                            <div class='fk-det-card'>
                                <div class='fk-det-top'>
                                    <span class='fk-det-name'>{rec['object_label'].title()} · Frame {rec['frame_number']}</span>
                                    <span class='fk-conf-pill'>{rec['confidence_score']*100:.1f}%</span>
                                </div>
                                <div class='fk-muted' style='margin-top:4px;'>{rec['timestamp_formatted']} · {_dec_badge(cur_dec)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if orig_mp4 is not None and orig_mp4.exists():
                            frame_rgb = extract_annotated_frame_rgb(orig_mp4, rec["frame_number"], rec["norm_box"], rec["object_label"], rec["confidence_score"])
                            if frame_rgb is not None:
                                st.image(frame_rgb, width="stretch")

                        b1, b2, b3 = st.columns(3, gap="small")
                        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                        if b1.button("Accept", key=f"sum_acc_{vid_id}_{obs_i}", width="stretch"):
                            vid_reviews[rkey] = {"decision": "Accept", "reviewedAt": now_utc}
                            st.rerun()
                        if b2.button("Reject", key=f"sum_rej_{vid_id}_{obs_i}", width="stretch"):
                            vid_reviews[rkey] = {"decision": "Reject", "reviewedAt": now_utc}
                            st.rerun()
                        if b3.button("Uncertain", key=f"sum_unc_{vid_id}_{obs_i}", width="stretch"):
                            vid_reviews[rkey] = {"decision": "Uncertain", "reviewedAt": now_utc}
                            st.rerun()

                st.markdown(
                    "<div class='fk-muted' style='margin-top:18px;'>Counts are observations across frames; a weapon may appear in several frames.</div>",
                    unsafe_allow_html=True,
                )


def render_multicam_page(mc_scenes: list[dict]) -> None:
    ss = st.session_state
    if not mc_scenes:
        st.warning("No multi-camera scenes indexed yet.")
        return

    scene = next((s for s in mc_scenes if s["scene_key"] == ss.get("active_mc_key")), mc_scenes[0])
    c1, c2 = scene["cam1"], scene["cam2"]
    mc_thresh = ss["mc_conf_pct"] / 100.0
    mc_intel = ss["cctv_intel"]

    t1 = load_telemetry(c1["telemetry_rel"])
    t2 = load_telemetry(c2["telemetry_rel"])
    recs1 = filter_records(t1, mc_thresh, mc_intel)
    recs2 = filter_records(t2, mc_thresh, mc_intel)
    combined = sorted(
        [{**r, "camera": "CAM-01"} for r in recs1] + [{**r, "camera": "CAM-02"} for r in recs2],
        key=lambda x: (x["timestamp_seconds"], x["camera"]),
    )

    with st.container(key="fk_body_wrap"):
        col_side, col_work, col_sum = st.columns([0.92, 2.30, 1.02], gap="small")

        with col_side:
            with st.container(key="fk_side_panel"):
                st.markdown("<div class='fk-sec fk-sec-first'>CAMERA SOURCES</div>", unsafe_allow_html=True)
                keys = [s["scene_key"] for s in mc_scenes]
                sel_i = st.selectbox(
                    "Multi-Camera Pair",
                    range(len(mc_scenes)),
                    index=keys.index(scene["scene_key"]),
                    format_func=lambda i: mc_scenes[i]["title"],
                    label_visibility="collapsed",
                    key="mc_side_pick",
                )
                if mc_scenes[sel_i]["scene_key"] != scene["scene_key"]:
                    ss["active_mc_key"] = mc_scenes[sel_i]["scene_key"]
                    st.rerun()

                for cam_lbl, info in [("CAM-01", c1), ("CAM-02", c2)]:
                    om = info["orig_meta"]
                    st.markdown(
                        f"""
                        <div class='fk-evidence-card'>
                            <div class='fk-evidence-title'>{cam_lbl} · {info['source_filename']}</div>
                            <div class='fk-muted'>{om['width']}×{om['height']} · {om['fps']} fps · {om['duration_seconds']}s</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<div class='fk-sec'>VIEW MODE</div>", unsafe_allow_html=True)
                with st.container(key="fk_mc_toggle_box"):
                    m1, m2 = st.columns(2, gap="small")
                    if m1.button("Original", type=("primary" if ss["mc_view_mode"] == "Original" else "secondary"), width="stretch", key="mc_vm_orig"):
                        ss["mc_view_mode"] = "Original"
                        st.rerun()
                    if m2.button("Detected", type=("primary" if ss["mc_view_mode"] == "Detected" else "secondary"), width="stretch", key="mc_vm_det"):
                        ss["mc_view_mode"] = "Detected"
                        st.rerun()

                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; align-items:center; margin:12px 0 2px 0;'>
                        <span class='fk-sec' style='margin:0;'>CONFIDENCE THRESHOLD</span>
                        <span style='color:#22396f; font-size:12.5px; font-weight:700;'>{ss['mc_conf_pct']}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                new_mc_c = st.slider("MC Conf", 20, 90, ss["mc_conf_pct"], 5, label_visibility="collapsed", key="mc_conf_sl")
                if new_mc_c != ss["mc_conf_pct"]:
                    ss["mc_conf_pct"] = new_mc_c
                    st.rerun()

                st.markdown("<div class='fk-sec' style='margin-top:16px;'>TRAINED MODEL</div>", unsafe_allow_html=True)
                st.markdown("<div class='fk-muted'>Model 12 (Production Video SOTA)<br/>Multi-Camera Corroboration</div>", unsafe_allow_html=True)

        with col_work:
            with st.container(key="fk_workspace_panel"):
                st.markdown(
                    f"""
                    <div class='fk-ws-heading'>
                        <span class='fk-ws-title'>MULTI-CAMERA WORKSPACE · {scene['title'].upper()}</span>
                        <span class='fk-muted'>{'Detected evidence' if ss['mc_view_mode'] == 'Detected' else 'Original evidence'}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                vc1, vc2 = st.columns(2, gap="medium")
                with vc1:
                    st.markdown(f"<div class='fk-sec fk-sec-first'>CAM-01 · {len(recs1)} ALERTS</div>", unsafe_allow_html=True)
                    p1 = WEB_DATA_DIR / (c1["annotated_video_rel"] if ss["mc_view_mode"] == "Detected" else c1["original_video_rel"])
                    if p1.exists():
                        st.video(str(p1))
                with vc2:
                    st.markdown(f"<div class='fk-sec fk-sec-first'>CAM-02 · {len(recs2)} ALERTS</div>", unsafe_allow_html=True)
                    p2 = WEB_DATA_DIR / (c2["annotated_video_rel"] if ss["mc_view_mode"] == "Detected" else c2["original_video_rel"])
                    if p2.exists():
                        st.video(str(p2))

                with st.container(key="fk_obs_box"):
                    st.markdown("<div class='fk-sec fk-sec-first'>UNIFIED CHRONOLOGICAL MULTI-CAMERA TIMELINE</div>", unsafe_allow_html=True)
                    if combined:
                        df_mc = pd.DataFrame([
                            {
                                "Time": r["timestamp_formatted"],
                                "Camera": r["camera"],
                                "Object": r["object_label"].title(),
                                "Confidence": f"{r['confidence_score']*100:.1f}%",
                                "Frame": r["frame_number"],
                            }
                            for r in combined
                        ])
                        st.dataframe(df_mc, width="stretch", hide_index=True, height=205)
                    else:
                        st.dataframe(pd.DataFrame(columns=["Time", "Camera", "Object", "Confidence", "Frame"]), width="stretch", hide_index=True, height=150)

        with col_sum:
            with st.container(key="fk_summary_panel"):
                st.markdown("<div class='fk-sec fk-sec-first'>DETECTION SUMMARY</div>", unsafe_allow_html=True)
                hg_mc = sum(1 for r in combined if r["object_label"] == "handgun")
                kn_mc = sum(1 for r in combined if r["object_label"] == "knife")
                st.markdown(
                    f"""
                    <div class='fk-metrics-row'>
                        <div class='fk-metric-card'><div class='fk-metric-num'>{len(combined)}</div><div class='fk-metric-lbl'>DETECTIONS</div></div>
                        <div class='fk-metric-card'><div class='fk-metric-num'>{hg_mc}</div><div class='fk-metric-lbl'>HANDGUNS</div></div>
                        <div class='fk-metric-card'><div class='fk-metric-num'>{kn_mc}</div><div class='fk-metric-lbl'>KNIVES</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                corroborated = len(recs1) > 0 and len(recs2) > 0
                st.markdown(
                    f"""
                    <div class='fk-status-card'>
                        <div class='fk-status-title'>{'Dual-Camera Corroborated' if corroborated else 'Single-Angle / Clean Scene'}</div>
                        <div class='fk-muted'>CAM-01: {len(recs1)} alerts · CAM-02: {len(recs2)} alerts</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='fk-sec'>FORENSIC EXPORT</div>", unsafe_allow_html=True)
                mc_pdf = build_multicam_pdf_bytes(scene, recs1, recs2, mc_thresh, mc_intel)
                mc_csv = pd.DataFrame(combined).to_csv(index=False).encode("utf-8") if combined else b"camera,timestamp_formatted,frame_number,object_label,confidence_score\n"
                st.download_button("Multi-Camera PDF Report", data=mc_pdf, file_name=f"FORENSIKADA_{scene['scene_key']}_multicam.pdf", mime="application/pdf", type="primary", width="stretch")
                st.download_button("Combined Timeline (CSV)", data=mc_csv, file_name=f"FORENSIKADA_{scene['scene_key']}_combined.csv", mime="text/csv", width="stretch")


def render_evaluator_guide_page() -> None:
    st.markdown("<div style='padding:16px 22px;'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1.0], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("<div class='fk-sec fk-sec-first'>EXPERT EVALUATION GUIDE</div>", unsafe_allow_html=True)
            st.markdown(
                """
                Welcome to **FORENSIKADA** — Multi-Camera CCTV Forensic Weapon Detection System (**National University – Manila, CCIT**).

                1. Click **Import videos** in the top header (or use the left Evidence Source selector) to test Handgun, Knife, No-Weapon, or your own custom short clip (`≤25 MB, ≤15s`).
                2. Click **Enhance video** to compare Original vs. **BasicVSR++** enhanced CCTV streams.
                3. Switch **View Mode** (`Original` vs. `Detected`), adjust **Confidence Threshold**, and toggle **CCTV Intelligence Layer**.
                4. Click any row in **DETECTED OBJECTS** or click **Observation review** to verify frames (`Accept` / `Reject` / `Uncertain`).
                5. Click **Forensic report** or **Save video + report** in the top header to inspect and download the SHA-256 Forensic PDF.
                6. Switch to **Multi-camera** in the tab strip to review synchronized `CAM-01` & `CAM-02` corroboration.
                """
            )
            if GOOGLE_FORM_URL:
                st.link_button("Open Expert Evaluation Form", GOOGLE_FORM_URL, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("<div class='fk-sec fk-sec-first'>VERIFIED PRODUCTION VIDEO & STATIC SOTA BENCHMARKS (MODELS 12 & 13)</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class='fk-metrics-row'>
                    <div class='fk-metric-card'><div class='fk-metric-num'>99.4%</div><div class='fk-metric-lbl'>VIDEO PRECISION</div></div>
                    <div class='fk-metric-card'><div class='fk-metric-num'>85.6%</div><div class='fk-metric-lbl'>VIDEO F1 SCORE</div></div>
                    <div class='fk-metric-card'><div class='fk-metric-num'>100%</div><div class='fk-metric-lbl'>NO-WEAPON SPEC.</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                | Benchmark (Zero Leakage) | Model 10 | **Model 12 (Video SOTA)** | **Model 13 (Static/HN SOTA)** |
                | :--- | :---: | :---: | :---: |
                | **Real CCTV Video Precision** | 100.00% | **99.40%** | 68.48% |
                | **Real CCTV Video Recall** | 37.10% | **75.11% (+38.01 pp)** | 51.13% |
                | **Real CCTV Video F1 Score** | 54.13% | **85.57% (+31.44 pp)** | 58.55% |
                | **Real CCTV Video mAP@0.5** | 55.17% | **78.89% (+23.72 pp)** | 37.41% |
                | **No-Weapon Video Spec. (948 frames)** | 100.00% | **100.00% (0 FP)** | 98.10% |
                | **External 350 Static mAP@0.5** | 62.38% | 65.52% | **74.82% (+9.30 pp)** |
                | **External 350 Static F1 Score** | 69.66% | 70.86% | **74.92% (Record)** |
                | **Staged Tapo Val 44 (Post-CCTV)** | — | 37/44 | **44/44 (100.0%)** |
                """
            )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    index = load_library_index()
    single_videos = index.get("single_videos", [])
    mc_scenes = index.get("multicam_scenes", [])
    init_state(single_videos, mc_scenes)

    ss = st.session_state
    inject_exact_desktop_css(dark_mode=(ss["theme"] == "dark"))

    # Branded Forensikada splash / loading screen on initial open or browser refresh
    if not ss.get("splash_shown", False):
        splash_logo = _logo_data_uri()
        splash_img = f"<img class='fk-splash-logo' src='{splash_logo}' alt='Forensikada' />" if splash_logo else ""
        st.markdown(
            f"""
            <div class='fk-splash-overlay' role='status' aria-label='Loading Forensikada'>
                {splash_img}
                <div class='fk-splash-title'>Forensikada</div>
                <div class='fk-splash-sub'>CCTV FORENSIC VIDEO ANALYSIS</div>
                <div class='fk-splash-track'><div class='fk-splash-bar'></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ss["splash_shown"] = True

    entry, orig_mp4, annot_mp4, filtered_records, paired_entry = get_active_single_context(single_videos)
    analysis_ready = bool(ss.get("analysis_ready", False))

    # ===== 1. EXACT DESKTOP TOP HEADER BAR =====
    with st.container(key="fk_header_bar"):
        hc_brand, hc_imp, hc_cls, hc_rep, hc_sav, hc_set = st.columns([3.3, 1.05, 0.95, 1.05, 1.25, 0.42], gap="small")
        with hc_brand:
            logo_uri = _logo_data_uri()
            logo_html = f"<img class='fk-brand-logo' src='{logo_uri}' alt='Forensikada' />" if logo_uri else ""
            st.markdown(
                f"""
                <div class='fk-brand-wrap'>
                    {logo_html}
                    <div>
                        <div class='fk-brand-title'>Forensikada</div>
                        <div class='fk-brand-sub'>CCTV FORENSIC VIDEO ANALYSIS</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hc_imp:
            if st.button("Import videos", type="primary", width="stretch", key="hdr_import"):
                dialog_import_videos(single_videos, mc_scenes)
        with hc_cls:
            if st.button("Close videos", width="stretch", disabled=(entry is None), key="hdr_close"):
                ss["active_video_id"] = None
                ss["custom_result"] = None
                ss["analysis_ready"] = False
                ss["view_mode"] = "Original"
                ss["selected_obs_idx"] = 0
                st.rerun()
        with hc_rep:
            if st.button("Forensic report", width="stretch", disabled=(entry is None or not analysis_ready), key="hdr_report"):
                dialog_forensic_report(entry, filtered_records, ss["conf_pct"] / 100.0, ss["cctv_intel"])
        with hc_sav:
            if entry is not None and analysis_ready:
                vid_reviews = ss["reviews"].setdefault(entry["video_id"], {})
                pdf_b = build_single_video_pdf_bytes(entry, filtered_records, ss["conf_pct"] / 100.0, ss["cctv_intel"], vid_reviews)
                st.download_button(
                    "Save report (PDF)",
                    data=pdf_b,
                    file_name=f"FORENSIKADA_{entry['video_id']}_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="hdr_save_dl",
                )
            else:
                st.button("Save video + report", width="stretch", disabled=True, key="hdr_save_dis")
        with hc_set:
            if st.button("⚙", width="stretch", key="hdr_settings", help="System settings"):
                dialog_settings()

    # ===== 2. EXACT DESKTOP MODE TAB STRIP =====
    with st.container(key="fk_mode_bar"):
        mt1, mt2, mt3, mt_sp = st.columns([0.95, 1.05, 1.15, 5.5], gap="small")
        if mt1.button("Single video", type=("primary" if ss["active_mode"] == "Single video" else "secondary"), width="stretch", key="mode_single"):
            ss["active_mode"] = "Single video"
            st.rerun()
        if mt2.button("Multi-camera", type=("primary" if ss["active_mode"] == "Multi-camera" else "secondary"), width="stretch", key="mode_mc"):
            ss["active_mode"] = "Multi-camera"
            st.rerun()
        if mt3.button("Evaluator guide", type=("primary" if ss["active_mode"] == "Evaluator guide" else "secondary"), width="stretch", key="mode_guide"):
            ss["active_mode"] = "Evaluator guide"
            st.rerun()

    # ===== 3. ACTIVE DESKTOP WORKSPACE =====
    if ss["active_mode"] == "Single video":
        render_single_video_page(single_videos, mc_scenes, entry, orig_mp4, annot_mp4, filtered_records, paired_entry)
    elif ss["active_mode"] == "Multi-camera":
        render_multicam_page(mc_scenes)
    else:
        render_evaluator_guide_page()


if __name__ == "__main__":
    main()
