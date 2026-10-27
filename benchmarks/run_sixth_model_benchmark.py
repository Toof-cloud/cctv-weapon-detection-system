import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import run_full_pipeline

TEST_JOBS = [
    {
        "name": "Knife Normal Video (evaluation_video)",
        "video": ROOT / "samples" / "evaluation_video.mp4",
        "output": ROOT / "outputs" / "sixth_model_outputs" / "knife_detection_evaluationvideo",
    },
    {
        "name": "Knife CCTV Video (NEW_KNIFE_VIDEO_11s)",
        "video": ROOT / "samples" / "NEW_KNIFE_VIDEO_11s.mp4",
        "output": ROOT / "outputs" / "sixth_model_outputs" / "knife_detection_cctv_footage",
    },
    {
        "name": "Handgun Normal Video (handgun_test-video)",
        "video": ROOT / "samples" / "handgun_test-video.mp4",
        "output": ROOT / "outputs" / "sixth_model_outputs" / "handgun_detection_cctv_footage",
    },
    {
        "name": "Handgun Staged CCTV Video (CAM02_Scene_004)",
        "video": ROOT / "samples" / "CAM02_Scene_004.mp4",
        "output": ROOT / "outputs" / "sixth_model_outputs" / "handgun_detection_staged_dataset",
    },
]

MODEL_PATH = ROOT / "best_weapon_detector_sixth_model.pth"


def main():
    print("=" * 65)
    print("STARTING MODEL 6 QUADRANT VIDEO BENCHMARK")
    print(f"Model: {MODEL_PATH.name}")
    print("=" * 65)

    for idx, job in enumerate(TEST_JOBS, start=1):
        print(f"\n[{idx}/{len(TEST_JOBS)}] Processing: {job['name']}")
        print(f"Video:  {job['video']}")
        print(f"Output: {job['output']}")

        try:
            result = run_full_pipeline(
                video_path=str(job["video"]),
                output_dir=str(job["output"]),
                model_path=str(MODEL_PATH),
                confidence_threshold=0.50,
                analysis_fps=25,
                enable_enhancement=True,
            )
            print(f"Successfully processed {job['name']}")
            print(f"Summary: {result.get('summary', {})}")
        except Exception as exc:
            print(f"Error processing {job['name']}: {exc}")

    print("\n" + "=" * 65)
    print("ALL 4 MODEL 6 VIDEO BENCHMARKS COMPLETED!")
    print("Outputs stored in: outputs/sixth_model_outputs/")
    print("=" * 65)


if __name__ == "__main__":
    main()
