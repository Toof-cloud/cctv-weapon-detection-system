"""
Package a clean, self-contained staging folder (`hf_space_bundle/`) ready to push
directly to a Hugging Face Space (Docker SDK, CPU Basic 2 vCPU / 16 GB RAM).

Usage:
    python web_app/prepare_hf_space_bundle.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "hf_space_bundle"

HF_README = """---
title: FORENSIKADA CCTV Weapon Detection System
emoji: 🛡️
colorFrom: red
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
---

# FORENSIKADA — Multi-Camera CCTV Weapon Detection System (Model 12 Video SOTA)

Open-access forensic evaluation platform for the **FORENSIKADA** thesis system:
- **Model 12 (Production Video SOTA):** Faster R-CNN ResNet-50 FPN v2 + WiSE-FT (`alpha=0.30`) + CCTV Intelligence Layer (`99.40%` Video Precision, `85.57%` F1, `100.00%` No-Weapon Specificity)
- **Prepared Evaluation Library:** Instant H.264 playback across Handgun, Knife, Non-Weapon, and Multi-Camera (`CAM-01` & `CAM-02`) CCTV footage with dynamic confidence threshold slider, CCTV Intelligence toggle, Observation Review, and live Forensic PDF + CSV export
- **Live Custom Upload:** Test your own short CCTV clip (`<= 25 MB`, `<= 15 seconds`) directly against Model 12
"""

GITATTRIBUTES = """*.pth filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
"""


def copy_clean_tree(src: Path, dst: Path) -> None:
    def _ignore(_dir: str, names: list[str]) -> list[str]:
        return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]

    shutil.copytree(src, dst, ignore=_ignore)


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    # 1. Root Dockerfile, HF README.md, .gitattributes, and .streamlit/config.toml
    shutil.copy2(ROOT / "web_app" / "Dockerfile", BUNDLE / "Dockerfile")
    (BUNDLE / "README.md").write_text(HF_README, encoding="utf-8")
    (BUNDLE / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    st_cfg = ROOT / ".streamlit" / "config.toml"
    if st_cfg.exists():
        (BUNDLE / ".streamlit").mkdir(parents=True, exist_ok=True)
        shutil.copy2(st_cfg, BUNDLE / ".streamlit" / "config.toml")

    # 2. Core detection engine (app/services, app/utils, app/models, dataset_analysis/build_model.py, run_full_pipeline.py)
    for sub in ["services", "utils", "models"]:
        src_sub = ROOT / "app" / sub
        if src_sub.exists():
            copy_clean_tree(src_sub, BUNDLE / "app" / sub)
    (BUNDLE / "app" / "__init__.py").write_text("", encoding="utf-8")
    (BUNDLE / "dataset_analysis").mkdir(parents=True, exist_ok=True)
    (BUNDLE / "dataset_analysis" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(ROOT / "dataset_analysis" / "build_model.py", BUNDLE / "dataset_analysis" / "build_model.py")
    shutil.copy2(ROOT / "run_full_pipeline.py", BUNDLE / "run_full_pipeline.py")

    # 3. Web application, desktop brand assets & pre-indexed library
    (BUNDLE / "web_app").mkdir(parents=True, exist_ok=True)
    for fname in ["app.py", "pdf_export_helper.py", "requirements-web.txt"]:
        shutil.copy2(ROOT / "web_app" / fname, BUNDLE / "web_app" / fname)
    assets_src = ROOT / "web_app" / "assets"
    if assets_src.exists():
        copy_clean_tree(assets_src, BUNDLE / "web_app" / "assets")

    lib_src = ROOT / "web_app" / "library_data"
    if not (lib_src / "library_index.json").exists():
        raise FileNotFoundError(
            "web_app/library_data/library_index.json not found. Run `python web_app/build_web_library.py` first."
        )
    copy_clean_tree(lib_src, BUNDLE / "web_app" / "library_data")

    # 4. Production Model 12 checkpoint
    ckpt_src = ROOT / "best_weapon_detector_twelfth_model.pth"
    if not ckpt_src.exists():
        raise FileNotFoundError(f"Missing production checkpoint: {ckpt_src}")
    shutil.copy2(ckpt_src, BUNDLE / ckpt_src.name)

    # Compute summary size
    total_bytes = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
    file_count = sum(1 for p in BUNDLE.rglob("*") if p.is_file())
    print(f"[DONE] Hugging Face Space bundle ready at: {BUNDLE}")
    print(f"       Files: {file_count} | Total Size: {total_bytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
