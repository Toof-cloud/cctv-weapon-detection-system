"""Video adapter/UI checks. Synthetic predictions are confined to these tests."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys
sys.dont_write_bytecode = True
import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox, QPushButton, QSlider

import mockup_ui.model_bridge as bridge_module
import mockup_ui.observation_review as review_module
from mockup_ui.app import MainWindow, VideoEnhancementDialog
from mockup_ui.model_bridge import (
    MOCKUP_DIR, TEMP_DIR, ModelBridge, ModelSetupError, VideoAnalysisResult,
    VideoInputError, read_video, save_result,
)


def make_clip(path, frames=8, fps=5):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (240, 144))
    if not writer.isOpened():
        raise RuntimeError("Test MP4 writer unavailable")
    try:
        for number in range(frames):
            image = np.full((144, 240, 3), (35, 45, 55), dtype=np.uint8)
            cv2.putText(image, "TEST CLIP", (8, 120), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1)
            cv2.rectangle(image, (20 + number, 35), (85 + number, 80), (180, 160, 140), -1)
            writer.write(image)
    finally:
        writer.release()
    return read_video(path)


def fixture_result(video, detections=()):
    """A distinct processed test video lets UI tests check actual result playback."""
    output = video.path.parent / "TEST_PROCESSED.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), video.fps, (video.width, video.height))
    try:
        for number in range(video.frame_count):
            image = video.first_frame.copy()
            for row in detections:
                if row["frame_number"] != number:
                    continue
                x1, y1, x2, y2 = row["box"]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            writer.write(image)
    finally:
        writer.release()
    return VideoAnalysisResult(video, output, video.path.parent / "fixture.csv", list(detections),
                               "TEST_CHECKPOINT_ONLY.pth", "cpu", .1, .5, video.frame_count)


class VideoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_DIR)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.enterContext(patch.object(review_module, "REVIEW_DIR", self.folder / "reviews"))
        self.checkpoint = self.folder / "models" / bridge_module.MODEL_FILENAME
        self.checkpoint.parent.mkdir()
        self.enterContext(patch.object(bridge_module, "MODEL_PATH", self.checkpoint))
        self.video = make_clip(self.folder / "證據 TEST_CLIP.mp4")

    def test_preview_metadata_and_invalid_video_inputs(self):
        self.assertEqual(self.video.frame_count, 8)
        self.assertEqual((self.video.width, self.video.height), (240, 144))
        self.assertEqual(self.video.first_frame.shape, (144, 240, 3))
        self.assertAlmostEqual(self.video.duration, 1.6)
        self.assertAlmostEqual(self.video.fps, 5)
        with self.assertRaises(VideoInputError):
            read_video(self.folder / "missing.mp4")
        with self.assertRaises(VideoInputError):
            read_video(self.folder / "picture.jpg")
        bad = self.folder / "corrupt.mp4"
        bad.write_bytes(b"this is not a video")
        with self.assertRaises(VideoInputError):
            read_video(bad)

    def test_missing_checkpoint_and_changed_input(self):
        bridge = ModelBridge()
        with self.assertRaisesRegex(ModelSetupError, "Trained model not found"):
            bridge.analyze_video(self.video)
        self.video.path.write_bytes(b"changed")
        with self.assertRaisesRegex(VideoInputError, "changed"):
            bridge.analyze_video(self.video)

    def test_original_pipeline_produces_completed_video_and_detection_report(self):
        import torch
        import run_full_pipeline as pipeline
        from app.services.detection_service import DetectionService

        checkpoint = self.checkpoint
        checkpoint.write_text("TEST ONLY; never loaded as weights")
        bridge = ModelBridge()
        bridge.device = "cpu"
        frames_seen = []
        loader_calls = []

        def create_test_service(confidence_threshold, model_path):
            loader_calls.append((confidence_threshold, model_path))
            # Only loading is replaced; original detect_frame() and rendering run.
            service = object.__new__(DetectionService)
            service.device = torch.device("cpu")
            service.model_path = model_path
            service.categories = {1: "handgun", 2: "knife"}
            service.target_classes = {"handgun", "knife"}
            service.confidence_threshold = confidence_threshold

            def predictions(images):
                index = len(frames_seen)
                frames_seen.append(images[0].clone())
                label = 1 if index == 2 else 2
                score = .92 if index in (2, 4, 5) else .2
                return [{"boxes": torch.tensor([[20, 30, 95, 85]]),
                         "labels": torch.tensor([label]), "scores": torch.tensor([score])}]
            service.model = predictions
            return service

        bridge._prepare_runtime = lambda progress: pipeline.detect_video_with_model

        source_hash = hashlib.sha256(self.video.path.read_bytes()).hexdigest()
        progress = []
        with patch.object(bridge_module, "TEMP_DIR", self.folder), \
             patch.object(pipeline, "DetectionService", side_effect=create_test_service), redirect_stdout(io.StringIO()):
            result = bridge.analyze_video(self.video, .5, lambda text, value: progress.append(value))
        self.assertEqual(loader_calls, [(.5, checkpoint)])
        self.assertEqual(result.model_path, str(checkpoint))
        self.assertEqual(len(frames_seen), self.video.frame_count)
        self.assertEqual(tuple(frames_seen[0].shape), (3, 144, 240))
        np.testing.assert_allclose(frames_seen[0][:, 0, 0].numpy(), self.video.first_frame[0, 0, ::-1] / 255)
        self.assertEqual([row["frame_number"] for row in result.detections], [2, 4, 5])
        self.assertEqual([row["timestamp_seconds"] for row in result.detections], [.4, .8, 1.0])
        self.assertEqual(result.counts, {"handgun": 1, "knife": 2})
        self.assertEqual(result.positive_frames, 3)
        self.assertEqual(result.analyzed_frames, 8)
        self.assertEqual(progress[-1], 100)
        self.assertTrue(any(0 < percent < 99 for percent in progress))
        self.assertEqual(hashlib.sha256(self.video.path.read_bytes()).hexdigest(), source_hash)
        self.assertEqual(read_video(result.output_path).frame_count, 8)
        capture = cv2.VideoCapture(str(result.output_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, rendered = capture.read()
        capture.release()
        self.assertTrue(ok)
        red = rendered[:, :, 2].astype(int)
        green = rendered[:, :, 1].astype(int)
        self.assertTrue(np.any((red > 150) & (red > green + 70)), "Expected original renderer's red detection box")
        saved = save_result(result, self.folder)
        self.assertTrue((saved / "annotated.mp4").is_file())
        self.assertEqual((saved / "detections.csv").read_bytes(), result.csv_path.read_bytes())
        summary = json.loads((saved / "summary.json").read_text())
        self.assertEqual(summary["model_path"], str(checkpoint))
        self.assertEqual(summary["total_frame_detections"], 3)
        self.assertEqual(summary["detections"], result.detections)
        self.assertNotEqual(saved, save_result(result, self.folder))

    def test_partial_processing_is_not_reported_as_success(self):
        bridge = ModelBridge()
        bridge._prepare_runtime = lambda progress: lambda **kwargs: {"total_frames": 7, "analyzed_frames": 7}
        with patch.object(bridge_module, "TEMP_DIR", self.folder):
            with self.assertRaisesRegex(VideoInputError, "ended before"):
                bridge.analyze_video(self.video)

    def test_ninth_model_is_exclusive_despite_other_weights_and_legacy_settings(self):
        for name in ("best_weapon_detector.pth", "best_weapon_detector_retrained.pth"):
            (self.checkpoint.parent / name).write_text("TEST ONLY; never loaded as weights")
        other = self.checkpoint.parent / "best_weapon_detector_retrained.pth"
        (self.folder / "settings.json").write_text(json.dumps({"model_path": str(other)}))
        with patch.object(bridge_module, "MOCKUP_DIR", self.folder), \
             patch.dict(os.environ, {"FORENSIKADA_MODEL_PATH": str(other)}):
            bridge = ModelBridge()
            self.assertFalse(bridge.find_model())
            self.checkpoint.write_text("TEST ONLY; never loaded as weights")
            self.assertTrue(bridge.find_model())
            self.assertEqual(bridge.model_path, self.checkpoint)
            self.assertEqual(ModelBridge().model_path, self.checkpoint)
            with self.assertRaises(ModelSetupError):
                bridge.set_model_path(other)
            self.assertEqual(bridge.model_path, self.checkpoint)
            self.checkpoint.unlink()
            self.assertFalse(bridge.find_model())
            self.assertIsNone(bridge.model_path)
            with self.assertRaisesRegex(ModelSetupError, bridge_module.MODEL_FILENAME):
                bridge.analyze_video(self.video)

    def test_runtime_rechecks_required_checkpoint_before_loading(self):
        self.checkpoint.write_text("TEST ONLY; never loaded as weights")
        bridge = ModelBridge()
        self.assertEqual(bridge.model_path, self.checkpoint)
        other = self.folder / "best_weapon_detector.pth"
        other.write_text("TEST ONLY; never loaded as weights")
        self.checkpoint.unlink()
        bridge.model_path = other
        with self.assertRaisesRegex(ModelSetupError, bridge_module.MODEL_FILENAME):
            bridge.analyze_video(self.video)
        self.assertIsNone(bridge.model_path)


class VideoUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])
        cls.application.setStyle("Fusion")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_DIR)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.enterContext(patch.object(review_module, "REVIEW_DIR", self.folder / "reviews"))
        self.checkpoint = self.folder / "models" / bridge_module.MODEL_FILENAME
        self.checkpoint.parent.mkdir()
        self.enterContext(patch.object(bridge_module, "MODEL_PATH", self.checkpoint))
        self.video = make_clip(self.folder / "TEST_CLIP.mp4")
        self.checkpoint.write_text("TEST ONLY; never loaded as weights")
        self.window = MainWindow()
        self.window.show()
        QTest.qWait(30)
        self.addCleanup(self.close_window)
        self.errors = []
        self.window._show_error = lambda title, text: self.errors.append((title, text))

    def wait_finished(self):
        deadline = time.monotonic() + 10
        while self.window._thread and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window._thread)

    def close_window(self):
        self.wait_finished()
        if self.window._config_dialog:
            self.window._config_dialog.reject()
        self.window.close()

    def confirm_detection(self, threshold=None):
        dialog = self.window._config_dialog
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.isVisible())
        if threshold is not None:
            dialog.slider.setValue(threshold)
        dialog.accept()
        QTest.qWait(20)

    def test_import_never_starts_detection_cancel_preserves_video_and_manual_start_waits_for_model(self):
        self.checkpoint.unlink()
        self.window.bridge.analyze_video = lambda v, t, p: fixture_result(v)
        self.assertFalse(self.window.enhancement_button.isEnabled())
        self.assertIn("Import a video", self.window.enhancement_status.text())
        with patch.object(QFileDialog, "getOpenFileName") as dialog:
            self.window.load_video(str(self.video.path))
            dialog.assert_not_called()
        self.assertEqual(self.window.video_info.frame_count, 8)
        self.assertIsNone(self.window._thread)
        self.assertFalse(self.window._model_retry.isActive())
        self.assertEqual(self.window.status_title.text(), "Ready to analyze")
        self.assertTrue(self.window.analyze_button.isEnabled())
        self.assertEqual(self.window._config_dialog.slider.value(), 50)
        self.assertIn("No enhanced video has been created", self.window._config_dialog.enhancement_notice.text())
        self.assertIn("BasicVSR++ enhancement", self.window.enhancement_status.text())
        self.assertTrue(self.window.enhancement_button.isEnabled())
        self.window._config_dialog.reject()
        QTest.qWait(20)
        enhancement = VideoEnhancementDialog(self.video, self.window)
        self.assertFalse(enhancement.source_preview.pixmap().isNull())
        self.assertIn("BasicVSR++ output", enhancement.enhanced_preview.text())
        self.assertFalse(enhancement.run_button.isEnabled())
        self.assertEqual(enhancement.findChildren(QSlider), [])
        visible_copy = " ".join(label.text() for label in enhancement.findChildren(QLabel))
        self.assertIn("No manual adjustments required", visible_copy)
        self.assertNotIn("ENGINE NOT CONNECTED", visible_copy)
        enhancement.close()
        with patch("mockup_ui.app.VideoEnhancementDialog.exec", return_value=0) as preview:
            self.window.enhancement_button.click()
            preview.assert_called_once()
        self.assertEqual(self.window.source_path, self.video.path)
        self.assertIsNone(self.window.result)
        self.assertIn("remains imported", self.window.status_detail.text())
        self.assertFalse(self.window.player.play_button.isEnabled())
        self.window.show_detection_configuration()
        self.confirm_detection(63)
        self.assertEqual(self.window.threshold_slider.value(), 63)
        self.assertIsNone(self.window._thread)
        self.assertTrue(self.window._model_retry.isActive())
        self.assertEqual(self.window.status_title.text(), "Waiting for trained model")
        (self.checkpoint.parent / "best_weapon_detector.pth").write_text("TEST ONLY")
        self.window.analyze()
        self.assertIsNone(self.window._thread)
        self.assertIsNone(self.window.result)
        self.assertIn(bridge_module.MODEL_FILENAME, self.window.status_detail.text())
        self.checkpoint.write_text("TEST ONLY; never loaded as weights")
        deadline = time.monotonic() + 5
        while self.window.result is None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.wait_finished()
        self.assertIsNotNone(self.window.result)
        self.assertEqual(self.window.bridge.model_path, self.checkpoint)
        self.assertFalse(self.window._model_retry.isActive())
        self.assertTrue(self.window.player.timer.isActive())
        self.assertIn("not applied to this run", self.window.enhancement_status.text())

    def test_confirmed_import_uses_third_checkpoint_enables_report_and_destination_export(self):
        (self.checkpoint.parent / "best_weapon_detector_retrained.pth").write_text("TEST ONLY")
        thresholds = []
        self.window.bridge.analyze_video = lambda v, t, p: thresholds.append(t) or fixture_result(v)
        with patch.object(QFileDialog, "getOpenFileName") as dialog:
            self.window.load_video(str(self.video.path))
            self.assertFalse(self.window.player.timer.isActive())
            self.assertIsNone(self.window._thread)
            self.confirm_detection(71)
            self.wait_finished()
            self.assertTrue(self.window.player.timer.isActive())
            self.assertEqual(self.window.player.path, self.window.result.output_path)
            self.assertNotEqual(self.window.player.path, self.video.path)
            self.assertEqual(self.window.bridge.model_path, self.checkpoint)
            dialog.assert_not_called()
        self.assertEqual(self.window.threshold_slider.value(), 71)
        self.assertEqual(thresholds, [.71])
        self.assertTrue(self.window.report_button.isEnabled())
        with patch("mockup_ui.app.ForensicReportDialog.exec", return_value=0) as report:
            self.window.show_forensic_report()
            report.assert_called_once()
        with patch.object(QFileDialog, "getExistingDirectory", return_value=""), patch("mockup_ui.app.save_result") as save:
            self.window.save_current_result()
            save.assert_not_called()
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(self.folder)), \
             patch("mockup_ui.app.save_result", return_value=self.folder / "saved") as save, \
             patch.object(QMessageBox, "information"):
            self.window.save_current_result()
            save.assert_called_once_with(self.window.result, self.folder)
        self.assertIn(Path(self.window.result.model_path).name, self.window.model_label.text())
        self.assertFalse(hasattr(self.window, "model_settings_action"))
        self.assertFalse(self.window.model_label.actions())
        self.assertNotIn("Select model", [b.text() for b in self.window.findChildren(QPushButton)])

    def test_scanning_has_no_playback_or_totals_until_pipeline_finishes(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        calls = []
        def controlled(video, threshold, progress):
            calls.append(video.path)
            progress("Scanning test video...", 25)
            gate.wait(3)
            return fixture_result(video)
        self.window.bridge.analyze_video = controlled
        self.window.load_video(str(self.video.path))
        self.confirm_detection()
        self.window.analyze()
        deadline = time.monotonic() + 2
        while self.window.progress.value() != 25 and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(self.window._thread)
        self.assertEqual(self.window.progress.value(), 25)
        self.assertFalse(self.window.player.play_button.isEnabled())
        self.assertFalse(self.window.open_button.isEnabled())
        self.assertFalse(self.window.analyze_button.isEnabled())
        self.assertFalse(self.window.report_button.isEnabled())
        self.assertIsNotNone(self.window._processing_dialog)
        self.assertTrue(self.window._processing_dialog.isVisible())
        self.assertIn("frame 2 of 8", self.window._processing_dialog.frames.text().lower())
        self.assertEqual(self.window.total_metric.text(), "—")
        self.assertEqual(self.window.observation_model.rowCount(), 0)
        self.assertIsNone(self.window.result)
        self.window.player.play()
        QTest.qWait(250)
        self.assertEqual(self.window.player.frame_number, 0)
        self.assertFalse(self.window.player.timer.isActive())
        with patch.object(QMessageBox, "information"):
            self.assertFalse(self.window.close())
        gate.set()
        self.wait_finished()
        self.assertEqual(self.window.status_title.text(), "No weapons detected")
        self.assertEqual(self.window.total_metric.text(), "0")
        self.assertTrue(self.window.player.timer.isActive())
        self.assertTrue(self.window.save_button.isEnabled())
        self.assertTrue(self.window.report_button.isEnabled())
        self.assertIsNone(self.window._processing_dialog)
        self.assertEqual(self.window.player.path, self.window.result.output_path)

    def test_completed_summary_stays_constant_while_reviewing_boxed_frames(self):
        records = [
            {"frame_number": 2, "timestamp_seconds": .4, "class_name": "knife",
             "confidence": .92, "box": [20, 30, 95, 85]},
            {"frame_number": 4, "timestamp_seconds": .8, "class_name": "handgun",
             "confidence": .85, "box": [25, 30, 100, 90]},
        ]
        self.window.bridge.analyze_video = lambda v, t, p: fixture_result(v, records)
        self.window.load_video(str(self.video.path))
        self.confirm_detection()
        self.wait_finished()
        self.assertTrue(self.window.player.timer.isActive())
        self.assertEqual(self.window.total_metric.text(), "2")
        self.assertEqual(self.window.knife_metric.text(), "1")
        self.assertEqual(self.window.handgun_metric.text(), "1")
        self.assertEqual(self.window.observation_model.rowCount(), 2)
        self.window._jump_to_detection(self.window.observation_model.index(0, 0))
        self.assertEqual(self.window.player.frame_number, 2)
        self.assertIn("1 detection", self.window.frame_message.text())
        pixel = self.window.canvas._source_pixmap.toImage().pixelColor(20, 30)
        self.assertGreater(pixel.red(), pixel.green() + 70)
        self.assertEqual(self.window.results_layout.count(), 2)
        self.window.player.seek_frame(0)
        self.assertIn("No detections", self.window.frame_message.text())
        self.assertEqual(self.window.results_layout.count(), 1)
        self.assertEqual(self.window.total_metric.text(), "2")
        self.window.player.play()
        QTest.qWait(250)
        self.assertGreater(self.window.player.frame_number, 0)
        self.assertEqual(self.window.total_metric.text(), "2")
        self.window.player.pause()
        self.window.bridge.analyze_video = lambda *args: (_ for _ in ()).throw(RuntimeError("test failure"))
        self.window.load_video(str(self.video.path))
        self.confirm_detection()
        self.assertEqual(self.window.observation_model.rowCount(), 0)
        self.wait_finished()
        self.assertTrue(self.errors)
        self.assertIsNone(self.window.result)
        self.assertFalse(self.window.save_button.isEnabled())
        self.assertFalse(self.window.report_button.isEnabled())
        self.assertIsNone(self.window._processing_dialog)
        self.assertFalse(self.window.player.play_button.isEnabled())
        self.assertEqual(self.window.total_metric.text(), "—")
        self.assertFalse(self.window.player.timer.isActive())


if __name__ == "__main__":
    unittest.main()
