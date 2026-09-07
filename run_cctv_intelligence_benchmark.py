import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil
from run_full_pipeline import detect_video_with_model, generate_forensic_summary

TEST_JOBS = [
    {
        "name": "Handgun Staged CCTV Video (CAM02_Scene_004)",
        "source_video": ROOT / "samples" / "CAM02_Scene_004.mp4",
        "pre_enhanced": ROOT / "outputs" / "sixth_model_outputs" / "handgun_detection_staged_dataset" / "enhanced_video.mp4",
        "output_dir": ROOT / "outputs" / "cctv_intelligence_outputs" / "handgun_detection_staged_dataset",
    },
    {
        "name": "Knife CCTV Video (NEW_KNIFE_VIDEO_11s)",
        "source_video": ROOT / "samples" / "NEW_KNIFE_VIDEO_11s.mp4",
        "pre_enhanced": ROOT / "outputs" / "sixth_model_outputs" / "knife_detection_cctv_footage" / "enhanced_video.mp4",
        "output_dir": ROOT / "outputs" / "cctv_intelligence_outputs" / "knife_detection_cctv_footage",
    },
    {
        "name": "Knife Normal Video (evaluation_video)",
        "source_video": ROOT / "samples" / "evaluation_video.mp4",
        "pre_enhanced": ROOT / "outputs" / "sixth_model_outputs" / "knife_detection_evaluationvideo" / "enhanced_video.mp4",
        "output_dir": ROOT / "outputs" / "cctv_intelligence_outputs" / "knife_detection_evaluationvideo",
    },
    {
        "name": "Handgun Normal Video (handgun_test-video)",
        "source_video": ROOT / "samples" / "handgun_test-video.mp4",
        "pre_enhanced": ROOT / "outputs" / "sixth_model_outputs" / "handgun_detection_cctv_footage" / "enhanced_video.mp4",
        "output_dir": ROOT / "outputs" / "cctv_intelligence_outputs" / "handgun_detection_cctv_footage",
    },
]

MODEL_PATH = ROOT / "best_weapon_detector_sixth_model.pth"


def run_benchmark():
    print("=" * 70)
    print("STARTING CCTV SURVEILLANCE INTELLIGENCE BENCHMARK")
    print("Features: Centroid Motion Tracker + Person Proximity Gate + Geometric Area Filter")
    print(f"Model: {MODEL_PATH.name}")
    print("=" * 70)

    for idx, job in enumerate(TEST_JOBS, start=1):
        print(f"\n[{idx}/{len(TEST_JOBS)}] Processing: {job['name']}")
        out_dir = job["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy over pre-enhanced video if available
        enh_video_path = out_dir / "enhanced_video.mp4"
        if job["pre_enhanced"].exists() and not enh_video_path.exists():
            shutil.copy2(job["pre_enhanced"], enh_video_path)

        # 2. Run detection on unenhanced source video
        print(f"  -> Running detection on unenhanced video: {job['source_video'].name}")
        det_no_enh = detect_video_with_model(
            input_path=job["source_video"],
            output_path=out_dir / "annotated_output_no_enhancement.mp4",
            model_path=MODEL_PATH,
            confidence_threshold=0.50,
            analysis_fps=25,
            save_detected_frames=True,
            csv_output_path=out_dir / "forensic_detections_no_enhancement.csv",
            enable_cctv_intelligence=True,
        )
        generate_forensic_summary(
            Path(det_no_enh["csv_report"]),
            out_dir / "forensic_summary_no_enhancement.txt",
        )

        # 3. Run detection on enhanced video if present
        if enh_video_path.exists():
            print(f"  -> Running detection on enhanced video: {enh_video_path.name}")
            det_enh = detect_video_with_model(
                input_path=enh_video_path,
                output_path=out_dir / "annotated_output_enhanced.mp4",
                model_path=MODEL_PATH,
                confidence_threshold=0.50,
                analysis_fps=25,
                save_detected_frames=True,
                csv_output_path=out_dir / "forensic_detections_enhanced.csv",
                enable_cctv_intelligence=True,
            )
            generate_forensic_summary(
                Path(det_enh["csv_report"]),
                out_dir / "forensic_summary_enhanced.txt",
            )

    print("\n" + "=" * 70)
    print("ALL 4 CCTV INTELLIGENCE BENCHMARKS COMPLETED!")
    print("Outputs stored in: outputs/cctv_intelligence_outputs/")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
