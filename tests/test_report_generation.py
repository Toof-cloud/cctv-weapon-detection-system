from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.report_service import ReportService


def test_reporting():
    out_dir = ROOT / "outputs" / "test_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    service = ReportService(output_dir=out_dir)

    dummy_records = [
        {
            "source_video": "CAM02_Scene_004.mp4",
            "frame_number": 65,
            "timestamp_seconds": 3.98,
            "camera_id": "CAM-01",
            "object_label": "handgun",
            "box": [528, 518, 615, 669],
            "confidence_score": 0.8540,
            "validation_status": "VALIDATED_TEMPORAL",
            "rejection_reason": "",
        },
        {
            "source_video": "CAM02_Scene_004.mp4",
            "frame_number": 75,
            "timestamp_seconds": 4.59,
            "camera_id": "CAM-01",
            "object_label": "handgun",
            "box": [585, 637, 685, 720],
            "confidence_score": 0.6050,
            "validation_status": "VALIDATED_TEMPORAL",
            "rejection_reason": "",
        },
        {
            "source_video": "NEW_KNIFE_VIDEO_11s.mp4",
            "frame_number": 12,
            "timestamp_seconds": 0.48,
            "camera_id": "CAM-02",
            "object_label": "knife",
            "box": [1058, 642, 1279, 806],
            "confidence_score": 0.8220,
            "validation_status": "VALIDATED_TEMPORAL",
            "rejection_reason": "",
        },
        {
            "source_video": "CAM02_Scene_004.mp4",
            "frame_number": 20,
            "timestamp_seconds": 1.20,
            "camera_id": "CAM-01",
            "object_label": "knife",
            "box": [100, 200, 300, 400],
            "confidence_score": 0.7200,
            "validation_status": "SUPPRESSED",
            "rejection_reason": "STATIC_BACKGROUND_TRAP (Staircase handrail)",
        },
    ]

    case_meta = {
        "case_id": "TEST-CCTV-001",
        "analyst_name": "Tyrone & Howard",
        "cameras": [
            {
                "camera_id": "CAM-01",
                "filename": "CAM02_Scene_004.mp4",
                "resolution": "1280x720",
                "fps": 30.0,
                "duration_seconds": 6.8,
                "total_frames": 204,
            },
            {
                "camera_id": "CAM-02",
                "filename": "NEW_KNIFE_VIDEO_11s.mp4",
                "resolution": "1920x1080",
                "fps": 25.0,
                "duration_seconds": 11.0,
                "total_frames": 275,
            },
        ],
    }

    csv_path = service.export_csv(dummy_records, out_dir / "test_forensic_detections.csv")
    json_path = service.export_json(dummy_records, case_meta, out_dir / "test_forensic_audit.json")
    html_path = service.export_html_report(dummy_records, case_meta, out_dir / "test_forensic_report.html")
    pdf_path = service.export_pdf_report(dummy_records, case_meta, out_dir / "test_forensic_report.pdf")

    print(f"CSV generated:  {csv_path.exists()} ({csv_path.stat().st_size} bytes)")
    print(f"JSON generated: {json_path.exists()} ({json_path.stat().st_size} bytes)")
    print(f"HTML generated: {html_path.exists()} ({html_path.stat().st_size} bytes)")
    print(f"PDF generated:  {pdf_path.exists()} ({pdf_path.stat().st_size} bytes)")

    # Check 8 required columns in CSV
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        print("CSV Header columns:", header)
        required = [
            "source_video",
            "frame_number",
            "timestamp_seconds",
            "camera_id",
            "object_label",
            "bounding_box",
            "confidence_score",
            "validation_status",
        ]
        for col in required:
            assert col in header, f"Missing required column: {col}"
        print("ALL 8 REQUIRED FORENSIC COLUMNS VERIFIED IN CSV!")


if __name__ == "__main__":
    test_reporting()
