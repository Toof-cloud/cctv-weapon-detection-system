"""
Batch Enhancement Script
========================
Runs BasicVSR++ enhancement on every video in the given input folder and saves
results to mockup_ui/outputs/enhanced_videos/.

Usage (from the repo root):
    .venv\Scripts\python.exe batch_enhance.py
    .venv\Scripts\python.exe batch_enhance.py --input "path/to/folder" --output "path/to/output"
    .venv\Scripts\python.exe batch_enhance.py --dry-run
"""

import argparse
import sys
import time
from hashlib import sha256
from pathlib import Path

# ── repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))          # so `app.*` imports work

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = (
    REPO_ROOT
    / "samples"
    / "DEFENSE VIDEOS"
    / "To Enhance"
)
DEFAULT_OUTPUT = REPO_ROOT / "mockup_ui" / "outputs" / "enhanced_videos"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# ── helpers ───────────────────────────────────────────────────────────────────

def build_output_path(video_path: Path, output_dir: Path) -> Path:
    """Mirror the naming convention used by VideoEnhancementDialog in app.py."""
    stat       = video_path.stat()
    source_key = f"{video_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    digest     = sha256(source_key.encode("utf-8")).hexdigest()[:12]
    return output_dir / f"{video_path.stem}_{digest}_enhanced.mp4"


def fmt_size(path: Path) -> str:
    mb = path.stat().st_size / (1024 * 1024)
    return f"{mb:.1f} MB"


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-enhance CCTV videos with BasicVSR++ via WSL."
    )
    parser.add_argument(
        "--input", "-i",
        default=str(DEFAULT_INPUT),
        help="Folder containing source videos (default: samples/DEFENSE VIDEOS/To Enhance)",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help="Folder where enhanced videos are saved (default: mockup_ui/outputs/enhanced_videos)",
    )
    parser.add_argument(
        "--skip-existing", "-s",
        action="store_true",
        default=True,
        help="Skip videos whose enhanced output already exists (default: True)",
    )
    parser.add_argument(
        "--no-skip",
        dest="skip_existing",
        action="store_false",
        help="Re-enhance even if output already exists",
    )
    parser.add_argument(
        "--gpu-fraction", "-g",
        type=float,
        default=0.5,
        help="Fraction of GPU memory allocation to use (default: 0.5 for 50%%)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be done without actually enhancing anything",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    # ── validate input ────────────────────────────────────────────────────────
    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        sys.exit(1)

    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"[WARN] No video files found in: {input_dir}")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── verify WSL (skip during dry-run) ─────────────────────────────────────
    if not args.dry_run:
        print("Verifying WSL / BasicVSR++ environment...", flush=True)
        from app.services.video_enhancement_service import VideoEnhancementService
        enhancer = VideoEnhancementService()
        if not enhancer.verify_wsl_access():
            print(
                "[ERROR] WSL is not accessible. Make sure WSL is installed and "
                "the BasicVSR++ environment exists at /home/pc/thesis/mmagic_env/"
            )
            sys.exit(1)
        print("WSL OK\n")
    else:
        enhancer = None

    # ── print plan ────────────────────────────────────────────────────────────
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Videos : {len(videos)} found")
    print(f"GPU Cap: {int(args.gpu_fraction * 100)}% VRAM allocation")
    print(f"Mode   : {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print("-" * 60)

    skipped = 0
    processed = 0
    failed_list: list[tuple[Path, str]] = []

    for idx, video in enumerate(videos, 1):
        out_path = build_output_path(video, output_dir)
        prefix   = f"[{idx:>2}/{len(videos)}]"

        if args.skip_existing and out_path.exists():
            print(f"{prefix} SKIP  {video.name}  ->  already enhanced")
            skipped += 1
            continue

        print(f"{prefix} START {video.name}  ({fmt_size(video)})")
        print(f"        -> {out_path.name}")

        if args.dry_run:
            print(f"        [DRY-RUN - skipping actual enhancement]\n")
            continue

        t0 = time.time()
        try:
            enhancer.enhance_video(str(video), str(out_path), gpu_fraction=args.gpu_fraction)

            elapsed = time.time() - t0
            if out_path.exists():
                print(
                    f"        Done in {fmt_elapsed(elapsed)}  "
                    f"({fmt_size(out_path)})\n"
                )
                processed += 1
            else:
                msg = "Output file was not created after enhancement."
                print(f"        FAILED: {msg}\n")
                failed_list.append((video, msg))

        except Exception as exc:
            elapsed = time.time() - t0
            msg = str(exc)
            print(f"        FAILED after {fmt_elapsed(elapsed)}: {msg}\n")
            failed_list.append((video, msg))

    # ── summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    if args.dry_run:
        print(f"DRY-RUN complete - {len(videos)} videos would be processed.")
    else:
        print(f"Done.  Enhanced: {processed}  |  Skipped: {skipped}  |  Failed: {len(failed_list)}")
        if failed_list:
            print("\nFailed videos:")
            for vid, reason in failed_list:
                print(f"  - {vid.name}: {reason}")


if __name__ == "__main__":
    main()
