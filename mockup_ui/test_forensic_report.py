"""Report checks use explicitly synthetic saved records, never real predictions."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mockup_ui.forensic_report import MOCKUP_DIR, write_forensic_report


class ForensicReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=MOCKUP_DIR / "temp")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / "TEST_ONLY.mp4"
        self.source.write_bytes(b"Synthetic source reference for report tests; not decoded")
        self.row = {"frame_number": 13, "timestamp_seconds": .54, "class_name": "knife",
                    "confidence": .8072, "box": [329, 316, 353, 456]}
        self.summary = {
            "source_video": str(self.source), "model_path": "TEST_SAVED_MODEL_REFERENCE.pth",
            "device": "cpu", "confidence_threshold": .5, "fps": 24.0,
            "duration_seconds": 6.041666667, "analyzed_frames": 145,
            "elapsed_seconds": 12.3, "detections": [self.row],
        }
        self.summary_path = self.folder / "summary.json"
        self.write_saved_run()

    def write_saved_run(self):
        self.summary_path.write_text(json.dumps(self.summary))
        with (self.folder / "detections.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_number", "timestamp_seconds", "class_name", "confidence", "x1", "y1", "x2", "y2"])
            for row in self.summary["detections"]:
                writer.writerow([row["frame_number"], row["timestamp_seconds"], row["class_name"], row["confidence"], *row["box"]])

    def test_report_preserves_saved_model_values_and_marks_missing_metadata_and_review(self):
        original = self.summary_path.read_bytes()
        html = write_forensic_report(self.summary_path)
        report = json.loads(html.with_suffix(".json").read_text())
        row = report["detections"][0]
        self.assertEqual(row["model_reference"], "TEST_SAVED_MODEL_REFERENCE.pth")
        self.assertEqual(row["source_video_reference"], str(self.source))
        self.assertEqual(row["frame_number"], 13)
        self.assertEqual(row["video_relative_timestamp_seconds"], .54)
        self.assertEqual([row[key] for key in ("x1", "y1", "x2", "y2")], self.row["box"])
        self.assertEqual(row["confidence_score"], .8072)
        self.assertIsNone(row["source_timestamp"])
        self.assertIsNone(row["camera_id"])
        self.assertIsNone(row["reviewer"])
        self.assertEqual(row["validation_status"], "pending_human_validation")
        self.assertIsNone(report["processing"]["completed_at_utc"])
        self.assertEqual(report["source"]["file_at_report_generation"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        with (html.parent / "forensic_records.csv").open(encoding="utf-8-sig", newline="") as stream:
            exported = list(csv.DictReader(stream))
        self.assertEqual(exported[0]["record_id"], row["record_id"])
        self.assertEqual(exported[0]["camera_id"], "")
        self.assertEqual(exported[0]["camera_id_status"], "not_recorded_by_current_pipeline")
        self.assertNotIn("automated_validation_status", exported[0])
        rendered = html.read_text(encoding="utf-8")
        self.assertIn("80.72%", rendered)
        self.assertIn("This is a system-generated report and analyst review is provided here.", rendered)
        self.assertNotIn("Automated Validation Result", rendered)
        headings = ("Video Metadata", "Video and Detection Information", "Interpretation",
                    "Object Detection Observations and Reviews", "Analyst Review Information",
                    "Source References", "Traceability Report")
        positions = [rendered.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.summary_path.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            write_forensic_report(self.summary_path, html.parent)

    def test_disagreement_between_csv_and_summary_blocks_report(self):
        self.summary["detections"][0]["confidence"] = .99
        self.summary_path.write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(ValueError, "disagree"):
            write_forensic_report(self.summary_path)

    def test_zero_detections_missing_source_and_html_escaping(self):
        self.source.unlink()
        self.summary["detections"] = []
        self.summary["model_path"] = "<script>TEST_ONLY</script>"
        self.write_saved_run()
        html = write_forensic_report(self.summary_path)
        report = json.loads(html.with_suffix(".json").read_text())
        self.assertEqual(report["summary"]["total_frame_detections"], 0)
        self.assertEqual(report["source"]["file_at_report_generation"]["status"], "unavailable")
        self.assertIn("No handgun or knife observations met", html.read_text())
        self.assertNotIn("<script>", html.read_text())
        with (html.parent / "forensic_records.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertIn("validation_status", reader.fieldnames)
            self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
