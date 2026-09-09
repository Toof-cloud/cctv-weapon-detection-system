"""Review persistence, decision rules, and report integration with explicit fixtures."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide6.QtPdf import QPdfDocument

import mockup_ui.observation_review as review_module
from mockup_ui.observation_review import ReviewStore, validate_review
from mockup_ui.review_panel import ObservationReviewDialog
from mockup_ui.model_bridge import TEMP_DIR, result_summary, save_result
from mockup_ui.test_mockup import make_clip, fixture_result
from mockup_ui.app import MainWindow, ForensicReportDialog
from mockup_ui.forensic_report import write_forensic_report


class ObservationReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_DIR)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.enterContext(patch.object(review_module, "REVIEW_DIR", self.folder / "reviews"))
        self.video = make_clip(self.folder / "TEST_REVIEW_ONLY.mp4")
        self.rows = [
            {"frame_number": 2, "timestamp_seconds": .4, "class_name": "knife", "confidence": .92,
             "box": [20, 30, 95, 85], "automated_validation_status": "validated"},
            {"frame_number": 4, "timestamp_seconds": .8, "class_name": "handgun", "confidence": .85,
             "box": [25, 30, 100, 90]},
        ]
        self.result = fixture_result(self.video, self.rows)
        self.store = ReviewStore.for_result(self.result)

    def panel(self):
        panel = ObservationReviewDialog(self.result)
        panel.show()
        QTest.qWait(20)
        self.addCleanup(self.close_panel, panel)
        return panel

    def close_panel(self, panel):
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard):
            panel.close()

    def click(self, panel, decision):
        QTest.mouseClick(panel.decision_buttons[decision], Qt.MouseButton.LeftButton)

    def test_rules_and_invalid_attempts_never_change_saved_file(self):
        before = self.store.path.read_bytes()
        identifier = self.store.observations[0]["observationId"]
        for decision, notes in ((None, ""), ("Reject", ""), ("Reject", " \t\n"), ("Other", "notes")):
            with self.subTest(decision=decision, notes=notes):
                with self.assertRaises(ValueError):
                    self.store.save_review(identifier, decision, notes)
                self.assertEqual(self.store.path.read_bytes(), before)
        for decision in ("Accept", "Uncertain"):
            self.assertEqual(validate_review(decision, " \t\n"), "")

    def test_choices_are_exclusive_manual_and_required_notes_follow_current_decision(self):
        panel = self.panel()
        self.assertEqual([b.text() for b in panel.decision_group.buttons()], ["Accept", "Reject", "Uncertain"])
        self.assertIsNone(panel.decision_group.checkedButton())
        self.assertIn("validated", panel.automated_label.text())
        panel.save_review()
        self.assertEqual(panel.feedback.text(), "Please select Accept, Reject, or Uncertain before completing the review.")
        self.click(panel, "Reject")
        panel.notes.setPlainText(" \n\t")
        panel.save_review()
        self.assertEqual(panel.feedback.text(), "A reason is required when rejecting an observation.")
        self.assertEqual(panel.notes_label.text(), "Reason for Rejection *")
        self.click(panel, "Accept")
        self.assertEqual(sum(b.isChecked() for b in panel.decision_group.buttons()), 1)
        self.assertNotIn("*", panel.notes_label.text())
        panel.notes.clear()
        panel.save_review()
        self.assertEqual(panel.store.observations[0]["analystReview"]["decision"], "Accept")
        self.click(panel, "Reject")
        panel.save_review()
        self.assertIn("reason is required", panel.feedback.text())
        self.assertEqual(panel.store.observations[0]["analystReview"]["decision"], "Accept")
        panel.notes.setPlainText("  Evidence does not support the observation.  ")
        panel.save_review()
        saved = panel.store.observations[0]
        self.assertEqual(saved["analystReview"]["notes"], "Evidence does not support the observation.")
        self.assertEqual(saved["automatedValidationStatus"], "validated")
        self.click(panel, "Uncertain")
        panel.notes.clear()
        panel.save_review()
        self.assertEqual(panel.store.observations[0]["analystReview"]["decision"], "Uncertain")
        self.assertEqual(panel.store.observations[0]["analystReview"]["notes"], "")

    def test_navigation_reopen_restore_and_new_run_keep_decisions_separate(self):
        panel = self.panel()
        self.click(panel, "Reject")
        panel.notes.setPlainText("Saved reason")
        panel.save_review()
        first = deepcopy(panel.store.observations[0])
        panel.observation_list.setCurrentRow(1)
        self.assertIsNone(panel.decision_group.checkedButton())
        self.assertEqual(panel.notes.toPlainText(), "")
        self.assertIn("Not performed", panel.automated_label.text())
        self.click(panel, "Uncertain")
        panel.save_review()
        panel.observation_list.setCurrentRow(0)
        self.assertTrue(panel.decision_buttons["Reject"].isChecked())
        self.assertEqual(panel.notes.toPlainText(), "Saved reason")
        panel.close()
        restored = ReviewStore(self.store.path).restore_result()
        self.assertEqual(result_summary(restored), result_summary(self.result))
        restored_store = ReviewStore.for_result(restored)
        self.assertEqual(restored_store.observations[0], first)
        reopened = self.panel()
        self.assertTrue(reopened.decision_buttons["Reject"].isChecked())
        self.assertEqual(reopened.notes.toPlainText(), "Saved reason")
        self.assertIn(first["analystReview"]["reviewedAt"], reopened.feedback.text())
        fresh = fixture_result(self.video, self.rows)
        self.assertIsNone(ReviewStore.for_result(fresh).observations[0]["analystReview"])
        self.assertNotEqual(fresh.run_id, restored.run_id)

    def test_failed_write_preserves_saved_review_and_ui_allows_retry(self):
        panel = self.panel()
        before = self.store.path.read_bytes()
        self.click(panel, "Accept")
        with patch.object(review_module, "atomic_json", side_effect=OSError("disk full")):
            panel.save_review()
        self.assertIn("could not be saved", panel.feedback.text())
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertIsNone(panel.store.observations[0]["analystReview"])
        panel.save_review()
        self.assertEqual(panel.store.observations[0]["analystReview"]["decision"], "Accept")

    def test_report_pdf_and_export_preserve_both_statuses_notes_source_and_timestamp(self):
        identifiers = [row["observationId"] for row in self.store.observations]
        saved = self.store.save_review(identifiers[0], "Reject", "The visible object is inconclusive. <not markup>")
        self.store.save_review(identifiers[1], "Uncertain", "")
        # The original pipeline CSV has no analyst data and remains byte-identical.
        import csv
        with self.result.csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_number", "timestamp_seconds", "class_name", "confidence", "x1", "y1", "x2", "y2"])
            for row in self.rows:
                writer.writerow([row["frame_number"], row["timestamp_seconds"], row["class_name"], row["confidence"], *row["box"]])
        original_csv = self.result.csv_path.read_bytes()
        exported = save_result(self.result, self.folder / "export")
        self.assertEqual((exported / "detections.csv").read_bytes(), original_csv)
        summary = json.loads((exported / "summary.json").read_text())
        self.assertEqual(summary["detections"], self.rows)
        report = json.loads((exported / "forensic_report.json").read_text())
        row = report["detections"][0]
        self.assertEqual(row["automated_validation_status"], "validated")
        self.assertEqual(row["analyst_decision"], "Reject")
        self.assertEqual(row["analyst_notes"], saved["notes"])
        self.assertEqual(row["observation_id"], identifiers[0])
        self.assertEqual(row["reviewed_at"], saved["reviewedAt"])
        self.assertEqual(report["observations"], self.store.observations)
        self.assertEqual(ReviewStore(exported / "observation_reviews.json").observations, self.store.observations)
        document = QPdfDocument(self.application)
        self.assertEqual(document.load(str(exported / "forensic_report.pdf")), QPdfDocument.Error.None_)
        text = "\n".join(document.getAllText(i).text() for i in range(document.pageCount()))
        for required in ("Automated Validation Result", "Analyst Review Decision", "Reject", "Uncertain", "validated", "<not markup>", saved["reviewedAt"]):
            self.assertIn(required, text)
        document.close()
        from shiboken6 import delete
        delete(document)  # Release Qt PDFium's Windows file handle before fixture cleanup.
        dialog = ForensicReportDialog(self.result)
        self.assertEqual(dialog.model.data(dialog.model.index(0, 5)), "validated")
        self.assertEqual(dialog.model.data(dialog.model.index(0, 6)), "Reject")
        self.assertEqual(dialog.model.data(dialog.model.index(0, 7)), saved["notes"])
        dialog.close()

    def test_reopen_saved_review_in_application_does_not_run_inference(self):
        self.store.save_review(self.store.observations[0]["observationId"], "Accept", "")
        window = MainWindow()
        try:
            with patch.object(QFileDialog, "getOpenFileName", return_value=(str(self.store.path), "")), \
                 patch.object(window, "show_observation_review") as opened, \
                 patch.object(window.bridge, "analyze_video") as inference:
                window.open_saved_review()
            inference.assert_not_called()
            opened.assert_called_once()
            self.assertEqual(window.result.run_id, self.result.run_id)
            self.assertTrue(window.review_button.isEnabled())
            self.assertTrue(window.report_button.isEnabled())
            self.assertEqual(ReviewStore.for_result(window.result).observations[0]["analystReview"]["decision"], "Accept")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
