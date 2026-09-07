import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.multi_camera_service import CameraInputConfig, MultiCameraService
from app.services.report_service import ReportService
from app.utils.file_validation import validate_video_file
from app.utils.timestamps import format_timestamp


class AnalysisWorker(QThread):
    """Background worker thread to run video analysis without blocking the GUI."""
    progress = Signal(str, int, int, float)  # camera_id, current_frame, total_frames, percent
    status_updated = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        camera_configs: List[CameraInputConfig],
        confidence_threshold: float = 0.50,
        analysis_fps: int = 5,
        enable_cctv_intelligence: bool = True,
        enable_temporal_consistency: bool = True,
        model_path: Optional[Path | str] = None,
    ):
        super().__init__()
        self.camera_configs = camera_configs
        self.confidence_threshold = confidence_threshold
        self.analysis_fps = analysis_fps
        self.enable_cctv_intelligence = enable_cctv_intelligence
        self.enable_temporal_consistency = enable_temporal_consistency
        self.model_path = model_path

    def run(self):
        try:
            self.status_updated.emit("Initializing Weapon Detector and CCTV Intelligence...")
            service = MultiCameraService(
                output_dir=ROOT_DIR / "outputs" / "ui_multicam_run",
                confidence_threshold=self.confidence_threshold,
                analysis_fps=self.analysis_fps,
                enable_cctv_intelligence=self.enable_cctv_intelligence,
                enable_temporal_consistency=self.enable_temporal_consistency,
                model_path=self.model_path,
            )

            def on_progress(cam_id, cur_f, tot_f, pct):
                self.progress.emit(cam_id, cur_f, tot_f, pct)
                self.status_updated.emit(f"Analyzing {cam_id}: Frame {cur_f}/{tot_f} ({pct:.1f}%)")

            results = service.process_cameras(
                camera_configs=self.camera_configs,
                analyst_name="Howard (UI) & Tyrone (Prototype)",
                progress_callback=on_progress,
            )
            self.status_updated.emit("Analysis and Forensic Report generation complete!")
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    FORENSIKADA - Multi-Camera CCTV Weapon Detection & Forensic Reporting System.
    Built for Howard (UI) and Tyrone (Prototype).
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("FORENSIKADA - Multi-Camera CCTV Weapon Detection & Forensic Analysis")
        self.resize(1280, 850)

        self.cam1_file: Optional[Path] = None
        self.cam2_file: Optional[Path] = None
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.all_records: List[Dict[str, Any]] = []

        self.create_interface()

    def create_interface(self):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title & Subtitle
        header_layout = QVBoxLayout()
        title = QLabel("FORENSIKADA: Forensic CCTV Multi-Camera Weapon Detection")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #0f172a;")
        subtitle = QLabel("Multi-Camera Surveillance Feed Analysis with Temporal Consistency & Forensic Audit Reporting")
        subtitle.setStyleSheet("font-size: 13px; color: #64748b;")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addLayout(header_layout)

        # Dual Camera Input Section
        cameras_layout = QHBoxLayout()

        # Camera 1 Group
        cam1_group = QGroupBox("Camera 1 (Primary Surveillance Feed)")
        cam1_group.setStyleSheet("font-weight: bold; color: #0369a1;")
        cam1_layout = QVBoxLayout(cam1_group)

        c1_id_layout = QHBoxLayout()
        c1_id_layout.addWidget(QLabel("Camera Identifier:"))
        self.cam1_id_input = QLineEdit("CAM-01")
        self.cam1_id_input.setFixedWidth(120)
        c1_id_layout.addWidget(self.cam1_id_input)
        c1_id_layout.addStretch()
        cam1_layout.addLayout(c1_id_layout)

        self.cam1_select_btn = QPushButton("Select Camera 1 Video File")
        self.cam1_select_btn.clicked.connect(self.select_cam1_video)
        self.cam1_file_label = QLabel("No video selected")
        self.cam1_file_label.setStyleSheet("color: #64748b; font-weight: normal;")
        cam1_layout.addWidget(self.cam1_select_btn)
        cam1_layout.addWidget(self.cam1_file_label)

        self.cam1_meta_label = QLabel("Resolution: N/A | FPS: N/A | Duration: N/A")
        self.cam1_meta_label.setStyleSheet("font-size: 11px; font-weight: normal; color: #475569;")
        cam1_layout.addWidget(self.cam1_meta_label)
        cameras_layout.addWidget(cam1_group)

        # Camera 2 Group
        cam2_group = QGroupBox("Camera 2 (Secondary Surveillance Feed)")
        cam2_group.setStyleSheet("font-weight: bold; color: #0369a1;")
        cam2_layout = QVBoxLayout(cam2_group)

        c2_id_layout = QHBoxLayout()
        c2_id_layout.addWidget(QLabel("Camera Identifier:"))
        self.cam2_id_input = QLineEdit("CAM-02")
        self.cam2_id_input.setFixedWidth(120)
        c2_id_layout.addWidget(self.cam2_id_input)
        c2_id_layout.addStretch()
        cam2_layout.addLayout(c2_id_layout)

        self.cam2_select_btn = QPushButton("Select Camera 2 Video File")
        self.cam2_select_btn.clicked.connect(self.select_cam2_video)
        self.cam2_file_label = QLabel("No video selected (Optional)")
        self.cam2_file_label.setStyleSheet("color: #64748b; font-weight: normal;")
        cam2_layout.addWidget(self.cam2_select_btn)
        cam2_layout.addWidget(self.cam2_file_label)

        self.cam2_meta_label = QLabel("Resolution: N/A | FPS: N/A | Duration: N/A")
        self.cam2_meta_label.setStyleSheet("font-size: 11px; font-weight: normal; color: #475569;")
        cam2_layout.addWidget(self.cam2_meta_label)
        cameras_layout.addWidget(cam2_group)

        main_layout.addLayout(cameras_layout)

        # Settings & Pipeline Options
        settings_layout = QHBoxLayout()

        self.cctv_intel_check = QCheckBox("CCTV Intelligence (Person Proximity + Static Trap Filter)")
        self.cctv_intel_check.setChecked(True)
        settings_layout.addWidget(self.cctv_intel_check)

        self.temporal_check = QCheckBox("Temporal Consistency (Tracklet Persistence & Smoothing)")
        self.temporal_check.setChecked(True)
        settings_layout.addWidget(self.temporal_check)

        settings_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Model 7 (Micro Anchors & Hard Negatives)",
            "Model 6 (High Recall / VIRAT)",
            "Model 5 (Balanced / High Precision)",
        ])
        settings_layout.addWidget(self.model_combo)

        settings_layout.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.20, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.50)
        settings_layout.addWidget(self.conf_spin)

        settings_layout.addWidget(QLabel("Sample FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 30)
        self.fps_spin.setValue(5)
        settings_layout.addWidget(self.fps_spin)

        settings_layout.addStretch()
        main_layout.addLayout(settings_layout)

        # Action Button & Progress
        action_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Start Multi-Camera Forensic Analysis")
        self.analyze_button.setStyleSheet(
            "background-color: #0284c7; color: white; font-weight: bold; font-size: 14px; padding: 8px 16px; border-radius: 4px;"
        )
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        action_layout.addWidget(self.analyze_button)
        main_layout.addLayout(action_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready. Please select at least one CCTV video.")
        self.status_label.setStyleSheet("color: #475569; font-weight: bold;")
        main_layout.addWidget(self.status_label)

        # Results & Filter Bar
        results_header = QHBoxLayout()
        r_label = QLabel("Traceable Forensic Detection Records (Chronological Multi-Camera Timeline)")
        r_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0f172a;")
        results_header.addWidget(r_label)
        results_header.addStretch()

        results_header.addWidget(QLabel("Filter Camera:"))
        self.cam_filter_combo = QComboBox()
        self.cam_filter_combo.addItems(["All Cameras", "CAM-01 Only", "CAM-02 Only"])
        self.cam_filter_combo.currentIndexChanged.connect(self.apply_table_filters)
        results_header.addWidget(self.cam_filter_combo)

        results_header.addWidget(QLabel("Status:"))
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItems(["All Records", "Confirmed Threats Only", "Suppressed Traps Only"])
        self.status_filter_combo.currentIndexChanged.connect(self.apply_table_filters)
        results_header.addWidget(self.status_filter_combo)

        main_layout.addLayout(results_header)

        # 8-Column Results Table (Exactly matching the required 8 fields)
        self.results_table = QTableWidget(0, 8)
        self.results_table.setHorizontalHeaderLabels([
            "Camera ID",
            "Timestamp",
            "Frame",
            "Weapon",
            "Confidence",
            "Bounding Box [x1, y1, x2, y2]",
            "Validation Status",
            "Source Video",
        ])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        main_layout.addWidget(self.results_table)

        # Forensic Export Action Buttons
        export_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export Forensic CSV Log")
        self.export_csv_btn.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.export_csv_btn.clicked.connect(self.export_csv_report)
        self.export_csv_btn.setEnabled(False)

        self.export_pdf_btn = QPushButton("Export Forensic PDF Incident Report")
        self.export_pdf_btn.setStyleSheet("font-weight: bold; background-color: #0f172a; color: white; padding: 6px 14px; border-radius: 4px;")
        self.export_pdf_btn.clicked.connect(self.export_pdf_report)
        self.export_pdf_btn.setEnabled(False)

        self.export_json_btn = QPushButton("Export Forensic JSON Audit File")
        self.export_json_btn.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.export_json_btn.clicked.connect(self.export_json_report)
        self.export_json_btn.setEnabled(False)

        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_pdf_btn)
        export_layout.addWidget(self.export_json_btn)
        export_layout.addStretch()
        main_layout.addLayout(export_layout)

        self.setCentralWidget(container)

    def select_cam1_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Camera 1 Video",
            str(ROOT_DIR / "samples"),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )
        if file_path:
            ok, err, meta = validate_video_file(file_path)
            if not ok:
                QMessageBox.critical(self, "Invalid Video File", err)
                return
            self.cam1_file = Path(file_path)
            self.cam1_file_label.setText(f"{self.cam1_file.name} ({meta['file_size_mb']} MB)")
            self.cam1_meta_label.setText(
                f"Resolution: {meta['resolution']} | FPS: {meta['fps']} | Duration: {meta['duration_seconds']}s ({meta['total_frames']} frames)"
            )
            self.analyze_button.setEnabled(True)
            self.status_label.setText(f"Status: Camera 1 loaded ({self.cam1_file.name}). Ready for analysis.")

    def select_cam2_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Camera 2 Video",
            str(ROOT_DIR / "samples"),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )
        if file_path:
            ok, err, meta = validate_video_file(file_path)
            if not ok:
                QMessageBox.critical(self, "Invalid Video File", err)
                return
            self.cam2_file = Path(file_path)
            self.cam2_file_label.setText(f"{self.cam2_file.name} ({meta['file_size_mb']} MB)")
            self.cam2_meta_label.setText(
                f"Resolution: {meta['resolution']} | FPS: {meta['fps']} | Duration: {meta['duration_seconds']}s ({meta['total_frames']} frames)"
            )
            self.status_label.setText(f"Status: Dual cameras configured (CAM-01 & CAM-02). Ready.")

    def start_analysis(self):
        if not self.cam1_file:
            QMessageBox.warning(self, "Missing Video", "Please select at least Camera 1 video.")
            return

        configs = [
            CameraInputConfig(
                camera_id=self.cam1_id_input.text().strip() or "CAM-01",
                video_path=self.cam1_file,
            )
        ]
        if self.cam2_file:
            configs.append(
                CameraInputConfig(
                    camera_id=self.cam2_id_input.text().strip() or "CAM-02",
                    video_path=self.cam2_file,
                )
            )

        self.analyze_button.setEnabled(False)
        self.cam1_select_btn.setEnabled(False)
        self.cam2_select_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results_table.setRowCount(0)
        self.all_records = []

        selected_model_idx = self.model_combo.currentIndex()
        if selected_model_idx == 1:
            chosen_model_path = ROOT_DIR / "best_weapon_detector_sixth_model.pth"
        elif selected_model_idx == 2:
            chosen_model_path = ROOT_DIR / "best_weapon_detector_fifth_model.pth"
        else:
            chosen_model_path = ROOT_DIR / "best_weapon_detector_seventh_model.pth"

        self.worker = AnalysisWorker(
            camera_configs=configs,
            confidence_threshold=self.conf_spin.value(),
            analysis_fps=self.fps_spin.value(),
            enable_cctv_intelligence=self.cctv_intel_check.isChecked(),
            enable_temporal_consistency=self.temporal_check.isChecked(),
            model_path=chosen_model_path,
        )
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.status_updated.connect(self.on_worker_status)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def on_worker_progress(self, cam_id, cur_f, tot_f, pct):
        self.progress_bar.setValue(int(pct))

    def on_worker_status(self, text):
        self.status_label.setText(f"Status: {text}")

    def on_worker_finished(self, results):
        self.analysis_results = results
        self.all_records = results.get("records", [])
        self.analyze_button.setEnabled(True)
        self.cam1_select_btn.setEnabled(True)
        self.cam2_select_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.export_pdf_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)

        self.populate_table(self.all_records)
        confirmed_count = results.get("confirmed_alerts", 0)
        total_recs = results.get("total_records", 0)

        QMessageBox.information(
            self,
            "Forensic Analysis Complete",
            f"Surveillance Video Analysis Completed Successfully!\n\n"
            f"Total Cameras Analyzed: {len(results.get('camera_results', {}))}\n"
            f"Total Evaluated Records: {total_recs}\n"
            f"Confirmed Weapon Threats: {confirmed_count}\n\n"
            f"Forensic PDF Report: {Path(results.get('unified_pdf', '')).name}\n"
            f"Forensic CSV Log: {Path(results.get('unified_csv', '')).name}",
        )

    def on_worker_error(self, err_msg):
        self.analyze_button.setEnabled(True)
        self.cam1_select_btn.setEnabled(True)
        self.cam2_select_btn.setEnabled(True)
        self.status_label.setText(f"Error: {err_msg}")
        QMessageBox.critical(self, "Analysis Failed", f"An error occurred during video analysis:\n{err_msg}")

    def populate_table(self, records: List[Dict[str, Any]]):
        self.results_table.setRowCount(0)
        for r in records:
            row_idx = self.results_table.rowCount()
            self.results_table.insertRow(row_idx)

            cam_item = QTableWidgetItem(r.get("camera_id", ""))
            cam_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row_idx, 0, cam_item)

            ts_item = QTableWidgetItem(f"{r.get('timestamp_formatted', '')} ({r.get('timestamp_seconds', 0.0)}s)")
            self.results_table.setItem(row_idx, 1, ts_item)

            frame_item = QTableWidgetItem(str(r.get("frame_number", "")))
            frame_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row_idx, 2, frame_item)

            weapon_label = str(r.get("object_label", "")).upper()
            weapon_item = QTableWidgetItem(weapon_label)
            weapon_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row_idx, 3, weapon_item)

            conf_item = QTableWidgetItem(f"{float(r.get('confidence_score', 0.0)):.1%}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row_idx, 4, conf_item)

            box_item = QTableWidgetItem(str(r.get("bounding_box", "")))
            self.results_table.setItem(row_idx, 5, box_item)

            status_text = r.get("validation_status", "CONFIRMED_ALERT")
            reason = r.get("rejection_reason", "")
            if reason:
                status_text += f" - {reason}"
            status_item = QTableWidgetItem(status_text)
            self.results_table.setItem(row_idx, 6, status_item)

            src_item = QTableWidgetItem(r.get("source_video", ""))
            self.results_table.setItem(row_idx, 7, src_item)

    def apply_table_filters(self):
        if not self.all_records:
            return

        cam_filter = self.cam_filter_combo.currentText()
        status_filter = self.status_filter_combo.currentText()

        filtered = []
        for r in self.all_records:
            cam_id = r.get("camera_id", "")
            if cam_filter == "CAM-01 Only" and cam_id != "CAM-01":
                continue
            if cam_filter == "CAM-02 Only" and cam_id != "CAM-02":
                continue

            val_status = r.get("validation_status", "")
            is_confirmed = val_status in ("CONFIRMED_ALERT", "VALIDATED_TEMPORAL")
            if status_filter == "Confirmed Threats Only" and not is_confirmed:
                continue
            if status_filter == "Suppressed Traps Only" and is_confirmed:
                continue

            filtered.append(r)

        self.populate_table(filtered)

    def export_csv_report(self):
        if not self.analysis_results:
            return
        csv_file = self.analysis_results.get("unified_csv")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Forensic CSV Log",
            f"forensic_detections_{Path(csv_file).name}",
            "CSV Files (*.csv)",
        )
        if save_path and csv_file and Path(csv_file).exists():
            import shutil
            shutil.copy2(csv_file, save_path)
            QMessageBox.information(self, "Export Successful", f"Forensic CSV report exported to:\n{save_path}")

    def export_pdf_report(self):
        if not self.analysis_results:
            return
        pdf_file = self.analysis_results.get("unified_pdf")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Forensic PDF Incident Report",
            f"forensic_report_{Path(pdf_file).name}",
            "PDF Files (*.pdf)",
        )
        if save_path and pdf_file and Path(pdf_file).exists():
            import shutil
            shutil.copy2(pdf_file, save_path)
            QMessageBox.information(self, "Export Successful", f"Forensic PDF Incident Report exported to:\n{save_path}")

    def export_json_report(self):
        if not self.analysis_results:
            return
        json_file = self.analysis_results.get("unified_json")
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Forensic JSON Audit Log",
            f"forensic_audit_{Path(json_file).name}",
            "JSON Files (*.json)",
        )
        if save_path and json_file and Path(json_file).exists():
            import shutil
            shutil.copy2(json_file, save_path)
            QMessageBox.information(self, "Export Successful", f"Forensic JSON audit log exported to:\n{save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())