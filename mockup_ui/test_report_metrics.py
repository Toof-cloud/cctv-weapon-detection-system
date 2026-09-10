"""Hand-calculated synthetic inputs for benchmark-backed report metrics."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mockup_ui.report_metrics import calculate_report_metrics
from mockup_ui.model_bridge import ModelBridge, VideoInfo, save_result


class ReportMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "metric_input.csv"

    def row(self, frame, status="VALIDATED_TEMPORAL", camera="CAM01", label="handgun", time=None):
        return {"frame_number": frame, "validation_status": status, "camera_id": camera,
                "object_label": label, "timestamp_seconds": frame if time is None else time,
                "source_video": str(self.folder / f"SYNTHETIC_{camera}.mp4"), "confidence_score": .8,
                "x1": 1, "y1": 2, "x2": 3, "y2": 4}

    def write_rows(self, rows, path=None):
        with (path or self.path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0] if rows else self.row(0)))
            writer.writeheader()
            writer.writerows(rows)

    def tcr_rows(self):
        return [self.row(0), self.row(1), self.row(2, "SUPPRESSED_TEMPORAL_FLICKER"),
                self.row(20), self.row(21)]

    def test_benchmark_counts_include_suppressed_rows_and_interruptions(self):
        self.write_rows(self.tcr_rows())
        metrics = calculate_report_metrics(self.path)
        tcr = metrics["tcr"]
        self.assertEqual([tcr[k] for k in ("n_ts", "n_isolated", "n_interrupted", "n_not_evaluable", "n_te")], [2, 1, 1, 2, 4])
        self.assertEqual(tcr["value_percent"], 50.0)
        self.assertIsNone(metrics["mccr"]["value_percent"])

    def test_two_camera_matching_and_conflicting_labels(self):
        self.write_rows([self.row(1), self.row(5, label="knife"), self.row(10),
                         self.row(1, camera="CAM02", time=1.4), self.row(5, camera="CAM02"),
                         self.row(10, camera="CAM02")])
        metrics = calculate_report_metrics(self.path)
        mccr = metrics["mccr"]
        self.assertEqual([mccr[k] for k in ("n_cc", "uncertain", "not_corroborated", "n_mc")], [4, 2, 0, 6])
        self.assertEqual(mccr["value_percent"], 66.67)
        self.assertEqual(metrics["tcr"]["status"], "per_video")
        self.assertEqual([v["n_not_evaluable"] for v in metrics["tcr"]["per_video"]], [2, 2])

    def test_missing_empty_and_ineligible_data_do_not_report_success(self):
        self.assertIsNone(calculate_report_metrics(self.path)["tcr"]["value_percent"])
        for rows in ([], [self.row(0)], [self.row(0), self.row(1, "not_performed"), self.row(2)]):
            self.write_rows(rows)
            self.assertIsNone(calculate_report_metrics(self.path)["tcr"]["value_percent"])
        self.write_rows([self.row(0, "SUPPRESSED_TEMPORAL_FLICKER"), self.row(0, "SUPPRESSED_TEMPORAL_FLICKER", camera="CAM02")])
        self.assertIsNone(calculate_report_metrics(self.path)["mccr"]["value_percent"])

    def test_bridge_preserves_raw_records_and_review_does_not_change_metrics(self):
        source = self.folder / "TEST_ONLY.mp4"
        source.write_bytes(b"synthetic source, no real prediction")
        stat = source.stat()
        video = VideoInfo(source, 20, 20, 1, 30, 30, None, stat.st_size, stat.st_mtime_ns)

        def pipeline(**kwargs):
            self.write_rows(self.tcr_rows(), kwargs["csv_output_path"])
            kwargs["output_path"].write_bytes(b"synthetic rendered video")
            return {"total_frames": 30, "analyzed_frames": 30}

        bridge = ModelBridge()
        bridge.device = "cpu"
        bridge.model_path = self.folder / "SYNTHETIC_MODEL.pth"
        with patch.object(bridge, "_prepare_runtime", return_value=pipeline), \
             patch("mockup_ui.model_bridge.read_video", return_value=video), \
             patch("mockup_ui.model_bridge.TEMP_DIR", self.folder):
            result = bridge.analyze_video(video, enable_temporal_consistency=True)
        self.assertEqual(len(result.detections), 4)
        original = result.metric_input_path.read_bytes()
        self.assertIn(b"SUPPRESSED_TEMPORAL_FLICKER", original)
        from mockup_ui.observation_review import ReviewStore
        store = ReviewStore.for_result(result)
        store.save_review(store.observations[0]["observationId"], "Reject", "Synthetic analyst decision")
        exported = save_result(result, self.folder / "export")
        self.assertEqual((exported / "metric_input.csv").read_bytes(), original)
        report = json.loads((exported / "forensic_report.json").read_text())
        self.assertEqual(report["metrics"]["tcr"]["value_percent"], 50.0)
        self.assertEqual(report["source"]["camera_id"], "CAM01")
        self.assertEqual(report["detections"][0]["analyst_decision"], "Reject")
        self.assertEqual(report["artifacts"]["metric_input"]["status"], "fingerprinted_at_report_generation")


if __name__ == "__main__":
    unittest.main()
