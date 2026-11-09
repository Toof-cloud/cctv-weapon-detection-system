"""
Auto-Boost Supervisor
====================
Monitors for the completion of Video 6, then smoothly transitions the remaining
videos (Videos 7 to 17) to 80% GPU allocation with larger batch chunks.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VIDEO6_ENHANCED = (
    REPO_ROOT
    / "mockup_ui"
    / "outputs"
    / "enhanced_videos"
    / "Screen Recording 2026-09-07 231101_61ba2404fef1_enhanced.mp4"
)

PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
BATCH_SCRIPT = REPO_ROOT / "batch_enhance.py"


def main():
    print("[AutoBoost] Waiting for Video 6 to finish before boosting to 80% GPU...", flush=True)

    while True:
        if VIDEO6_ENHANCED.exists() and VIDEO6_ENHANCED.stat().st_size > 1024 * 1024:
            print(f"[AutoBoost] Video 6 output verified: {VIDEO6_ENHANCED.name} ({VIDEO6_ENHANCED.stat().st_size / (1024*1024):.1f} MB)", flush=True)
            break
        time.sleep(10)

    # Allow 15 seconds for the current process to close files and clean up temp frames
    print("[AutoBoost] Allowing 15 seconds for file finalize and cleanup...", flush=True)
    time.sleep(15)

    # Terminate any running python batch_enhance process
    print("[AutoBoost] Terminating 50% baseline process...", flush=True)
    subprocess.run(
        ["powershell", "-Command", "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*batch_enhance.py*' -and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
        capture_output=True,
    )
    time.sleep(5)

    # Launch batch_enhance at 80% GPU memory allocation
    print("[AutoBoost] Launching Videos 7 to 17 at 80% GPU cap...", flush=True)
    cmd = [
        str(PYTHON_EXE),
        "-u",
        str(BATCH_SCRIPT),
        "--gpu-fraction",
        "0.8",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(f"[AutoBoost] 80% GPU batch pipeline launched (PID: {proc.pid}).", flush=True)
    proc.wait()


if __name__ == "__main__":
    main()
