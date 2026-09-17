"""Synthetic multi-camera integration checks; no test predictions enter production."""
import csv
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QPushButton

from mockup_ui.app import DetectionConfigDialog, MainWindow
from mockup_ui.model_bridge import VideoAnalysisResult, VideoInfo, read_video
from mockup_ui.multi_camera import Alignment, CameraSource, build_timeline, export_session, validate_session
from mockup_ui.multi_camera_panel import CameraPreviewDialog, CameraSetupDialog, MultiCameraDialog
from mockup_ui.observation_review import ReviewStore


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.sources = [CameraSource(VideoInfo(Path(f"TEST_CAM_{i}.mp4"), 320, 180, 10, 100, 10, None), f"CAM-{i}") for i in (1, 2)]
        self.alignment = Alignment("TEST-SCENE", True, "Synthetic shared clock", .25)

    def result(self, time, label="knife"):
        return SimpleNamespace(run_id=f"TEST-{time}-{label}", detections=[{
            "frame_number": round(time * 10), "timestamp_seconds": time, "class_name": label,
            "confidence": .8, "box": [1, 2, 3, 4], "automated_validation_status": "VALIDATED_TEMPORAL"}])

    def test_requires_correspondence_and_distinct_cameras(self):
        rows, metrics = build_timeline(self.sources, [self.result(1), self.result(1.1)], Alignment())
        self.assertTrue(all(row["corroboration_status"] == "Not Applicable" for row in rows))
        self.assertIsNone(metrics["mccr_percent"])
        self.sources[1].camera_id = self.sources[0].camera_id
        with self.assertRaisesRegex(ValueError, "different"):
            validate_session(self.sources, self.alignment)

    def test_offsets_enable_same_class_support_without_cross_view_iou(self):
        self.sources[1].offset_seconds = 2
        results = [self.result(3), self.result(1.1)]
        results[1].detections[0]["box"] = [200, 50, 300, 170]
        rows, metrics = build_timeline(self.sources, results, self.alignment)
        self.assertEqual(metrics["mccr_percent"], 100)
        self.assertEqual([row["session_seconds"] for row in rows], [3, 3.1])
        self.assertTrue(all(row["supporting_observations"] for row in rows))

    def test_conflicts_and_non_overlapping_footage_have_separate_outcomes(self):
        rows, metrics = build_timeline(self.sources, [self.result(1), self.result(1.1, "handgun")], self.alignment)
        self.assertEqual(metrics["counts"], {"Uncertain": 2})
        self.assertIsNone(metrics["mccr_percent"])
        self.sources[1].offset_seconds = 20
        rows, metrics = build_timeline(self.sources, [self.result(1), self.result(1.1)], self.alignment)
        self.assertEqual(metrics["counts"], {"Not Applicable": 2})

    def test_missing_support_is_zero_and_empty_detections_are_na(self):
        rows, metrics = build_timeline(self.sources, [self.result(1), self.result(8)], self.alignment)
        self.assertEqual(metrics["mccr_percent"], 0)
        rows, metrics = build_timeline(self.sources, [SimpleNamespace(run_id="A", detections=[]), SimpleNamespace(run_id="B", detections=[])], self.alignment)
        self.assertIsNone(metrics["mccr_percent"])


def make_video(path, caption):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 270))
    assert writer.isOpened()
    for frame in range(12):
        image = np.full((270, 480, 3), (40, 45, 58), dtype=np.uint8)
        cv2.putText(image, caption, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, .7, (220, 230, 240), 1)
        cv2.putText(image, "SYNTHETIC UI TEST", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, .65, (170, 185, 210), 1)
        cv2.rectangle(image, (180 + frame, 90), (270 + frame, 160), (75, 130, 190), 2)
        writer.write(image)
    writer.release()
    return read_video(path)


def fake_result(video, camera_id):
    folder = video.path.parent / camera_id
    folder.mkdir(exist_ok=True)
    output = folder / "annotated.mp4"
    output.write_bytes(video.path.read_bytes())
    detections = [{"frame_number": 2, "timestamp_seconds": .2, "class_name": "knife", "confidence": .8,
                   "box": [180, 90, 270, 160], "automated_validation_status": "VALIDATED_TEMPORAL"}]
    csv_path = folder / "detections.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_number", "timestamp_seconds", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerow([2, .2, "knife", .8, 180, 90, 270, 160])
    result = VideoAnalysisResult(video, output, csv_path, detections, "SYNTHETIC_MODEL.pth", "cpu", .1, .5, 12)
    result.review_path = folder / "reviews.json"
    return result


class MultiCameraUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent / "temp")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.videos = [make_video(self.folder / f"TEST_CAM_{i}.mp4", f"CAMERA {i}") for i in (1, 2)]
        self.window = MultiCameraDialog(self.videos)
        self.addCleanup(self.window.close)

    def test_batch_processing_playback_review_export_and_reset(self):
        self.window.scene.setText("TEST-SCENE")
        self.window.alignment.method = "Synthetic shared start"
        self.window.confirmed.setChecked(True)
        self.window.capture_settings()
        calls = []
        def analyze(video, threshold, progress, camera_id):
            calls.append(camera_id)
            progress("Synthetic fixture", 100)
            return fake_result(video, camera_id)
        self.window.bridge.analyze_video = analyze
        self.window.start_analysis()
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertFalse(self.window.setup.isEnabled())
        deadline = time.monotonic() + 5
        while self.window.worker and time.monotonic() < deadline:
            QTest.qWait(20)
        self.assertIsNone(self.window.worker)
        self.window.pause()
        self.assertEqual(calls, ["CAM-01", "CAM-02"])
        self.assertEqual(self.window.timeline.rowCount(), 2)
        self.assertIn("100.00%", self.window.metrics_label.text())
        self.assertEqual(self.window.cards[0][0].path, self.window.results[0].output_path)
        self.window.seek.setValue(200)
        self.assertEqual([card[0].frame_number for card in self.window.cards], [2, 2])
        store = ReviewStore.for_result(self.window.results[0])
        store.save_review(store.observations[0]["observationId"], "Reject", "Synthetic reason")
        exported = export_session(self.window.sources, self.window.results, self.window.alignment, self.folder)
        manifest = json.loads((exported / "session.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["metrics"]["mccr_percent"], 100)
        self.assertEqual(manifest["observations"][0]["analyst_decision"], "Reject")
        self.assertEqual(len(manifest["cameras"]), 2)
        self.window.configure()
        self.assertEqual(self.window.results, [])
        self.assertTrue(self.window.setup.isEnabled())
        self.assertFalse(self.window.export_button.isEnabled())

    def test_multi_file_picker_routes_to_workspace(self):
        main = MainWindow()
        with patch.object(QFileDialog, "getOpenFileNames", return_value=([str(v.path) for v in self.videos], "")), patch.object(main, "open_multi_camera") as opened:
            main.choose_video()
            opened.assert_called_once_with([str(v.path) for v in self.videos])
        main.close()

    def test_config_has_readable_controls_and_temporal_dependency(self):
        dialog = DetectionConfigDialog(self.videos[0], 60, videos=self.videos)
        self.assertEqual(dialog.slider.value(), 60)
        dialog.cctv_intel_checkbox.setChecked(False)
        self.assertFalse(dialog.temporal_checkbox.isChecked())
        self.assertFalse(dialog.temporal_checkbox.isEnabled())
        dialog.close()

    def test_simple_setup_and_preview_zoom_controls(self):
        button_texts = [button.text() for button in self.window.findChildren(QPushButton)]
        self.assertIn("Camera and timing details", button_texts)
        self.assertEqual(button_texts.count("Preview"), 2)
        self.assertNotIn("Maximize", button_texts)
        self.assertNotIn("Restore", button_texts)
        self.assertFalse(any(editor for editor in self.window.editors))
        details = CameraSetupDialog(self.window.sources, self.window.alignment, self.window)
        details.editors[0][1].setText("Entry view")
        details.editors[1][2].setValue(1.25)
        details.apply()
        self.assertEqual(self.window.sources[0].location, "Entry view")
        self.assertEqual(self.window.sources[1].offset_seconds, 1.25)
        details.close()
        preview = CameraPreviewDialog(self.videos[0].path, "TEST CAM", 2, self.window)
        preview.show()
        QTest.qWait(20)
        preview.player.canvas.set_zoom(2.0)
        self.assertEqual(preview.player.canvas.zoom, 2.0)
        QTest.mouseDClick(preview.player.canvas, Qt.MouseButton.LeftButton)
        self.assertEqual(preview.player.canvas.zoom, 1.0)
        preview_buttons = [button.text() for button in preview.findChildren(QPushButton)]
        self.assertEqual(preview_buttons, ["Play", "Start"])
        self.assertTrue(preview.player.start_button.isHidden())
        self.assertNotIn("Maximize", preview_buttons)
        self.assertNotIn("Restore", preview_buttons)
        preview.close()


if __name__ == "__main__":
    unittest.main()
