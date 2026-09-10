"""Forensikada video recognition and weapon detection interface."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import traceback

MOCKUP_DIR = Path(__file__).resolve().parent
ROOT_DIR = MOCKUP_DIR.parent
TEMP_DIR = MOCKUP_DIR / "temp"
for directory in (TEMP_DIR, MOCKUP_DIR / "outputs", MOCKUP_DIR / "models"):
    directory.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
# Give the original app/ package priority over this launcher's app.py filename.
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFontDatabase, QIcon, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QTableView,
    QHeaderView,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
)

from mockup_ui.model_bridge import (
    MODEL_FILENAME, ModelBridge, ModelSetupError, VideoAnalysisResult, VideoInfo, VideoInputError,
    discover_available_models, read_video, save_result, timecode,
)
from mockup_ui.video_player import VideoPlayer
from mockup_ui.observation_review import REVIEW_DIR, ReviewStore
from mockup_ui.review_panel import ObservationReviewDialog

logging.basicConfig(
    filename=TEMP_DIR / "mockup.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


class DetectionTableModel(QAbstractTableModel):
    HEADERS = ("Time", "Object", "Confidence", "Frame")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = []

    def set_records(self, records):
        self.beginResetModel()
        self.records = records
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self.records[index.row()]
        return (timecode(row["timestamp_seconds"]), row["class_name"].title(),
                f"{row['confidence']:.1%}", str(row["frame_number"]))[index.column()]


class ReportTableModel(QAbstractTableModel):
    HEADERS = ("Time", "Frame", "Object", "Confidence", "Bounding box", "Analyst decision", "Analyst notes", "Reviewed at (UTC)")

    def __init__(self, records, parent=None, observations=None):
        super().__init__(parent)
        self.records = records
        self.observations = observations or []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        row = self.records[index.row()]
        box = row["box"]
        observation = self.observations[index.row()] if self.observations else {}
        review = observation.get("analystReview") or {}
        return (timecode(row["timestamp_seconds"]), str(row["frame_number"]),
                row["class_name"].title(), f"{row['confidence']:.1%}",
                f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
                review.get("decision", "Not reviewed"), review.get("notes", ""), review.get("reviewedAt", ""))[index.column()]


class EnhancementFramePreview(QLabel):
    """Responsive still-frame preview used only by the enhancement setup UI."""

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        height, width, _ = frame.shape
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
        self._source_pixmap = QPixmap.fromImage(image.copy())
        self.setObjectName("enhancementPreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 230)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._fit()

    def _fit(self):
        if not self._source_pixmap.isNull():
            self.setPixmap(self._source_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event):
        self._fit()
        super().resizeEvent(event)


class VideoEnhancementDialog(QDialog):
    """Visual surface for the automatic BasicVSR++ enhancement stage."""

    def __init__(self, video: VideoInfo, parent=None):
        super().__init__(parent)
        self.setObjectName("videoEnhancementDialog")
        self.setWindowTitle("Video Enhancement")
        self.setModal(True)
        self.resize(1040, 660)
        self.setMinimumSize(880, 600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 22, 25, 20)
        layout.setSpacing(14)

        header = QHBoxLayout()
        heading_copy = QVBoxLayout()
        heading_copy.setSpacing(2)
        heading = QLabel("Video Enhancement")
        heading.setObjectName("dialogHeading")
        detail = QLabel(
            f"{video.path.name} · {video.width} × {video.height} · {video.fps:.2f} FPS · "
            f"{timecode(video.duration)}"
        )
        detail.setObjectName("dialogDetail")
        detail.setWordWrap(True)
        heading_copy.addWidget(heading)
        heading_copy.addWidget(detail)
        header.addLayout(heading_copy)
        header.addStretch()
        layout.addLayout(header)

        profile = QFrame()
        profile.setObjectName("basicVsrProfile")
        profile_layout = QHBoxLayout(profile)
        profile_layout.setContentsMargins(13, 9, 13, 9)
        profile_name = QLabel("BasicVSR++")
        profile_name.setObjectName("enhancementModel")
        profile_description = QLabel("Automatic video enhancement · No manual adjustments required")
        profile_description.setObjectName("enhancementStatus")
        profile_layout.addWidget(profile_name)
        profile_layout.addSpacing(8)
        profile_layout.addWidget(profile_description)
        profile_layout.addStretch()
        layout.addWidget(profile)

        body = QHBoxLayout()
        body.setSpacing(15)
        previews = QFrame()
        previews.setObjectName("enhancementPreviewArea")
        preview_layout = QHBoxLayout(previews)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(11)

        original_card = QFrame()
        original_card.setObjectName("enhancementPreviewCard")
        original_layout = QVBoxLayout(original_card)
        original_layout.setContentsMargins(10, 10, 10, 10)
        original_title = QLabel("SOURCE FRAME")
        original_title.setObjectName("enhancementPreviewTitle")
        original_layout.addWidget(original_title)
        self.source_preview = EnhancementFramePreview(video.first_frame)
        original_layout.addWidget(self.source_preview, 1)

        enhanced_card = QFrame()
        enhanced_card.setObjectName("enhancementPreviewCard")
        enhanced_layout = QVBoxLayout(enhanced_card)
        enhanced_layout.setContentsMargins(10, 10, 10, 10)
        enhanced_title = QLabel("ENHANCED PREVIEW")
        enhanced_title.setObjectName("enhancementPreviewTitle")
        enhanced_layout.addWidget(enhanced_title)
        self.enhanced_preview = QLabel(
            "Enhanced frame preview\n\nBasicVSR++ output will appear here"
        )
        self.enhanced_preview.setObjectName("enhancementPreviewPlaceholder")
        self.enhanced_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.enhanced_preview.setWordWrap(True)
        self.enhanced_preview.setMinimumSize(200, 230)
        enhanced_layout.addWidget(self.enhanced_preview, 1)
        preview_layout.addWidget(original_card, 1)
        preview_layout.addWidget(enhanced_card, 1)
        body.addWidget(previews, 1)
        layout.addLayout(body, 1)

        notice = QLabel(
            "UI preview only: BasicVSR++ enhancement is automatic and requires no manual settings. "
            "This mockup does not alter frames or detection input yet."
        )
        notice.setObjectName("enhancementNotice")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.run_button = buttons.addButton("Run BasicVSR++ enhancement", QDialogButtonBox.ButtonRole.AcceptRole)
        self.run_button.setObjectName("primaryButton")
        self.run_button.setEnabled(False)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class DetectionConfigDialog(QDialog):
    def __init__(self, video: VideoInfo, threshold: int, parent=None, current_model: Path | None = None):
        super().__init__(parent)
        self.setObjectName("configurationDialog")
        self.setWindowTitle("Detection configuration")
        self.setModal(True)
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 22)
        layout.setSpacing(14)
        heading = QLabel("Analyze imported video")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        detail = QLabel(
            f"{video.path.name}\n{video.width} × {video.height} · {video.fps:.2f} FPS · "
            f"{video.frame_count:,} frames · {timecode(video.duration)}"
        )
        detail.setObjectName("dialogDetail")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        layout.addSpacing(5)

        layout.addWidget(QLabel("DETECTION MODEL"))
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("dialogModelCombo")
        models = discover_available_models()
        active_idx = 0
        for idx, (label, path) in enumerate(models):
            self.model_combo.addItem(label, str(path))
            if current_model:
                try:
                    if Path(path).resolve() == Path(current_model).resolve():
                        active_idx = idx
                except Exception:
                    pass
        if models:
            self.model_combo.setCurrentIndex(active_idx)
        layout.addWidget(self.model_combo)

        layout.addSpacing(5)
        layout.addWidget(QLabel("CONFIDENCE THRESHOLD"))
        row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 95)
        self.slider.setValue(threshold)
        self.value_label = QLabel(f"{threshold}%")
        self.value_label.setObjectName("dialogThreshold")
        self.slider.valueChanged.connect(lambda value: self.value_label.setText(f"{value}%"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label)
        layout.addLayout(row)
        note = QLabel("Only detections meeting this confidence score will appear in the result. You can cancel and configure the video later.")
        note.setObjectName("dialogDetail")
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addSpacing(5)
        layout.addWidget(QLabel("FORENSIC CCTV FILTERS"))
        self.cctv_intel_checkbox = QCheckBox("Enable CCTV Intelligence (Kinematic & anthropometric scale gating)")
        self.cctv_intel_checkbox.setChecked(True)
        self.cctv_intel_checkbox.setToolTip("Suppresses false alarms when no person is present or when bounding box size violates human reach geometry.")
        layout.addWidget(self.cctv_intel_checkbox)

        self.temporal_checkbox = QCheckBox("Enable Temporal Consensus (Multi-frame trajectory verification)")
        self.temporal_checkbox.setChecked(True)
        self.temporal_checkbox.setToolTip("Suppresses isolated single-frame flickers and enforces temporal weapon trajectory persistence.")
        layout.addWidget(self.temporal_checkbox)

        if hasattr(video, "path") and "_enhanced" in video.path.name.lower():
            enh_text = (
                "VIDEO ENHANCEMENT\n"
                "BasicVSR++ enhanced video is active! Detection will analyze the super-resolved footage."
            )
        else:
            enh_text = (
                "VIDEO ENHANCEMENT\n"
                "No enhanced video has been created. Detection will use the imported video. "
                "Open Enhancement setup first if you want to run BasicVSR++ super-resolution."
            )
        self.enhancement_notice = QLabel(enh_text)
        self.enhancement_notice.setObjectName("enhancementNotice")
        self.enhancement_notice.setWordWrap(True)
        layout.addWidget(self.enhancement_notice)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        start = buttons.addButton("Start detection", QDialogButtonBox.ButtonRole.AcceptRole)
        start.setObjectName("primaryButton")
        start.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_model(self) -> Path | None:
        if hasattr(self, "model_combo") and self.model_combo.count() > 0:
            data = self.model_combo.currentData()
            if data:
                return Path(data)
        return None


class ProcessingDialog(QDialog):
    def __init__(self, total_frames: int, parent=None):
        super().__init__(parent)
        self.total_frames = total_frames
        self.setObjectName("processingDialog")
        self.setWindowTitle("Analyzing video")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setFixedWidth(500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 27, 30, 27)
        layout.setSpacing(12)
        heading = QLabel("Analyzing video…")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        self.message = QLabel("Preparing the trained detection model.")
        self.message.setObjectName("dialogDetail")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(True)
        self.progress.setMinimumHeight(18)
        layout.addWidget(self.progress)
        self.frames = QLabel(f"Preparing to process {total_frames:,} frames")
        self.frames.setObjectName("dialogDetail")
        layout.addWidget(self.frames)
        note = QLabel("Please wait while the system performs object detection and forensic analysis.")
        note.setObjectName("dialogDetail")
        note.setWordWrap(True)
        layout.addWidget(note)

    def update_progress(self, message: str, percent: int):
        self.message.setText(message)
        if percent < 0:
            self.progress.setRange(0, 0)
            self.frames.setText(f"Preparing to process {self.total_frames:,} frames")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
            current = min(self.total_frames, round(self.total_frames * percent / 100))
            self.frames.setText(f"Processing frame {current:,} of {self.total_frames:,} · {percent}% complete")

    def reject(self):
        pass  # Processing cannot be cancelled safely by the unchanged pipeline.


class ForensicReportDialog(QDialog):
    def __init__(self, result: VideoAnalysisResult, parent=None):
        super().__init__(parent)
        observations = ReviewStore.for_result(result).observations
        self.setObjectName("forensicReportDialog")
        self.setWindowTitle("Forensic Analysis Report")
        self.resize(980, 720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 21, 24, 20)
        layout.setSpacing(11)
        top = QHBoxLayout()
        heading_box = QVBoxLayout()
        heading = QLabel("Forensic Analysis Report")
        heading.setObjectName("dialogHeading")
        subtitle = QLabel(f"{result.video.path.name} · Generated {result.completed_at_utc}")
        subtitle.setObjectName("dialogDetail")
        heading_box.addWidget(heading)
        heading_box.addWidget(subtitle)
        top.addLayout(heading_box)
        top.addStretch()
        badge = QLabel(f"{sum(row['analystReview'] is not None for row in observations)} / {len(observations)} REVIEWED")
        badge.setObjectName("reviewBadge")
        top.addWidget(badge)
        layout.addLayout(top)

        disclaimer = QLabel("This is a system-generated report. All detections and validation results are subject to human analyst review and should be treated as reviewable observations, not conclusive findings.")
        disclaimer.setObjectName("reportObservation")
        disclaimer.setWordWrap(True)
        layout.addWidget(disclaimer)

        # Calculate TCR and MCCR metrics from metric_input.csv
        metric_path = getattr(result, "metric_input_path", None)
        if not metric_path or not Path(metric_path).exists():
            candidate = result.output_path.parent / "metric_input.csv"
            metric_path = candidate if candidate.exists() else None

        metrics = {}
        if metric_path:
            from mockup_ui.report_metrics import calculate_report_metrics
            metrics = calculate_report_metrics(Path(metric_path))

        tcr = metrics.get("tcr", {})
        mccr = metrics.get("mccr", {})

        if tcr.get("value_percent") is not None:
            tcr_str = (
                f"Rate: {tcr['value_percent']:.2f}%\n"
                f"Supported (N_TS): {tcr.get('n_ts', 0)} · Isolated: {tcr.get('n_isolated', 0)}\n"
                f"Total Eligible (N_TE): {tcr.get('n_te', 0)}"
            )
        elif tcr.get("status") == "per_video":
            tcr_str = "Per-Video Calculation\nSee detailed forensic log records"
        else:
            tcr_str = f"Status: N/A\n{tcr.get('reason', 'Pipeline records not available for this run')}"

        if mccr.get("value_percent") is not None:
            mccr_str = (
                f"Rate: {mccr['value_percent']:.2f}%\n"
                f"Corroborated (N_CC): {mccr.get('n_cc', 0)} / {mccr.get('n_mc', 0)} eligible"
            )
        else:
            mccr_str = (
                "Status: N/A (Single Camera Feed)\n"
                "Requires dual-camera layout with concurrent timestamps"
            )

        info = QGridLayout()
        values = (
            ("VIDEO INFORMATION", f"{result.video.width} × {result.video.height} · {result.video.fps:.2f} FPS\n"
                                  f"{timecode(result.video.duration)} · {result.video.frame_count:,} frames"),
            ("DETECTION CONFIGURATION", f"Threshold: {result.threshold:.0%}\nModel: {Path(result.model_path).name}"),
            ("DETECTION SUMMARY", f"{len(result.detections):,} observations · {result.positive_frames:,} positive frames\n"
                                  f"Handgun: {result.counts.get('handgun', 0):,} · Knife: {result.counts.get('knife', 0):,}"),
            ("PROCESSING", f"{result.analyzed_frames:,} frames analyzed on {result.device.upper()}\n"
                           f"Completed in {result.elapsed_seconds:.1f} seconds"),
            ("TEMPORAL CONSISTENCY (TCR)", tcr_str),
            ("MULTI-CAMERA CORROBORATION (MCCR)", mccr_str),
        )
        for index, (title, value) in enumerate(values):
            card = QFrame()
            card.setObjectName("reportCard")
            card_layout = QVBoxLayout(card)
            label = QLabel(title)
            label.setObjectName("sectionTitle")
            content = QLabel(value)
            content.setObjectName("dialogDetail")
            content.setWordWrap(True)
            card_layout.addWidget(label)
            card_layout.addWidget(content)
            info.addWidget(card, index // 2, index % 2)
        layout.addLayout(info)
        layout.addWidget(QLabel("DETECTION TIMELINE"))
        self.model = ReportTableModel(result.detections, self, observations)
        table = QTableView()
        table.setModel(self.model)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate((95, 55, 80, 90, 155, 120, 220, 210)):
            table.setColumnWidth(column, width)
        layout.addWidget(table, 1)
        summary = (f"Automated analysis recorded {len(result.detections):,} handgun/knife observation(s) "
                   f"across {result.positive_frames:,} frame(s). " if result.detections else
                   "No handgun or knife observation met the configured threshold. ")
        observation = QLabel(summary + "All observations, including rejected and uncertain analyst decisions, remain in this report. Camera ID and recording timestamps are not recorded.")
        observation.setObjectName("reportObservation")
        observation.setWordWrap(True)
        layout.addWidget(observation)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        layout.addWidget(close)


class InferenceWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str, str)
    progress = Signal(str, int)
    def __init__(self, bridge, video, threshold, parent=None):
        super().__init__(parent)
        self.bridge, self.video, self.threshold = bridge, video, threshold

    def run(self):
        try:
            self.succeeded.emit(self.bridge.analyze_video(self.video, self.threshold, self.progress.emit))
        except (VideoInputError, ModelSetupError) as exc:
            self.failed.emit(str(exc), traceback.format_exc())
        except Exception:
            self.failed.emit(
                "Video analysis could not finish. Check the trained checkpoint, video and available disk space. "
                "The original video remains available; technical details are in mockup_ui/temp/mockup.log.",
                traceback.format_exc(),
            )


class DetectionCard(QFrame):
    def __init__(self, detection: dict, index: int):
        super().__init__()
        self.setObjectName("detectionCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 11, 13, 11)
        layout.setSpacing(7)

        top = QHBoxLayout()
        name = QLabel(f"{index:02d}  {detection['class_name'].title()}")
        name.setObjectName("detectionName")
        score = QLabel(f"{detection['confidence']:.1%}")
        score.setObjectName("confidencePill")
        top.addWidget(name)
        top.addStretch()
        top.addWidget(score)
        layout.addLayout(top)
        box = detection["box"]
        coords = QLabel(f"Bounding box   [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
        coords.setObjectName("mutedText")
        coords.setWordWrap(True)
        layout.addWidget(coords)


class MainWindow(QMainWindow):
    def __init__(self, initial_video: str | None = None):
        super().__init__()
        if "Inter Variable" not in QFontDatabase.families():
            QFontDatabase.addApplicationFont(str(MOCKUP_DIR / "assets" / "InterVariable.ttf"))
        self.bridge = ModelBridge()
        self.source_path: Path | None = None
        self.video_info: VideoInfo | None = None
        self.result: VideoAnalysisResult | None = None
        self._detections_by_frame = {}
        self._thread: QThread | None = None
        self._analysis_requested = False
        self._config_dialog: DetectionConfigDialog | None = None
        self._processing_dialog: ProcessingDialog | None = None
        self._model_retry = QTimer(self)
        self._model_retry.setInterval(2000)
        self._model_retry.timeout.connect(self.analyze)
        self._build_ui()
        self._apply_style()
        self._update_model_label()
        self._set_ready("Import a CCTV video to begin.")
        if initial_video:
            self.load_video(initial_video)

    def _build_ui(self):
        self.setWindowTitle("Forensikada · Video Weapon Detection")
        self.resize(1450, 880)
        self.setMinimumSize(1040, 690)

        shell = QWidget()
        shell.setObjectName("shell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self._header())
        shell_layout.addWidget(self._body(), 1)
        self.setCentralWidget(shell)

    def _header(self):
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(22, 9, 18, 9)
        layout.setSpacing(10)
        logo = QLabel()
        logo.setFixedSize(44, 44)
        logo.setPixmap(QPixmap(str(MOCKUP_DIR / "assets" / "forensikada-logo.png")).scaled(
            44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        layout.addWidget(logo)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel("Forensikada")
        title.setObjectName("brandTitle")
        sub = QLabel("CCTV FORENSIC VIDEO ANALYSIS")
        sub.setObjectName("brandSub")
        brand.addWidget(title)
        brand.addWidget(sub)
        layout.addLayout(brand)
        layout.addStretch()

        self.open_button = QPushButton("Import video")
        self.open_button.setObjectName("primaryButton")
        self.open_button.setToolTip("Import a video and review detection settings before analysis.")
        self.open_button.setIcon(QIcon(str(MOCKUP_DIR / "assets" / "import-image.png")))
        self.open_button.clicked.connect(self.choose_video)
        self.report_button = QPushButton("Forensic report")
        self.report_button.setObjectName("headerButton")
        self.report_button.setToolTip("Review the completed forensic analysis inside the application.")
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.show_forensic_report)
        self.save_button = QPushButton("Save video + report")
        self.save_button.setToolTip("Save the annotated video, forensic report, and structured detection records.")
        self.save_button.setObjectName("headerButton")
        self.save_button.setIcon(QIcon(str(MOCKUP_DIR / "assets" / "save-result.svg")))
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current_result)
        for button in (self.open_button, self.report_button, self.save_button):
            button.setIconSize(QSize(18, 18))
            button.setMinimumHeight(42)
            layout.addWidget(button)
        return header

    def _body(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._sidebar())
        splitter.addWidget(self._workspace())
        splitter.addWidget(self._summary_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 850, 330])
        return splitter

    def _sidebar(self):
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(290)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 19, 18, 20)
        layout.setSpacing(11)
        layout.addWidget(self._section("EVIDENCE SOURCE"))
        evidence = QFrame()
        evidence.setObjectName("selectedEvidence")
        evidence_layout = QVBoxLayout(evidence)
        evidence_layout.setContentsMargins(13, 10, 13, 10)
        self.source_name = QLabel("No video selected")
        self.source_name.setTextFormat(Qt.TextFormat.PlainText)
        self.source_name.setObjectName("evidenceTitle")
        self.source_name.setWordWrap(True)
        self.source_meta = QLabel("Import CCTV footage")
        self.source_meta.setWordWrap(True)
        self.source_meta.setObjectName("mutedText")
        evidence_layout.addWidget(self.source_name)
        evidence_layout.addWidget(self.source_meta)
        layout.addWidget(evidence)

        layout.addSpacing(13)
        layout.addWidget(self._section("VIEW MODE"))
        toggle = QFrame()
        toggle.setObjectName("toggleBox")
        toggle_layout = QHBoxLayout(toggle)
        toggle_layout.setContentsMargins(4, 4, 4, 4)
        toggle_layout.setSpacing(0)
        self.original_button = QPushButton("Original")
        self.detected_button = QPushButton("Detected")
        self.original_button.setCheckable(True)
        self.detected_button.setCheckable(True)
        self.original_button.setChecked(True)
        self.detected_button.setEnabled(False)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_group.addButton(self.original_button)
        self.view_group.addButton(self.detected_button)
        self.original_button.clicked.connect(self.show_original)
        self.detected_button.clicked.connect(self.show_detected)
        toggle_layout.addWidget(self.original_button)
        toggle_layout.addWidget(self.detected_button)
        layout.addWidget(toggle)

        layout.addSpacing(13)
        layout.addWidget(self._section("CONFIDENCE THRESHOLD"))
        threshold_row = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 95)
        self.threshold_slider.setValue(50)
        self.threshold_value = QLabel("50%")
        self.threshold_value.setObjectName("thresholdValue")
        self.threshold_slider.valueChanged.connect(
            lambda value: self.threshold_value.setText(f"{value}%")
        )
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.threshold_value)
        layout.addLayout(threshold_row)
        self.analyze_button = QPushButton("Analyze video")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setToolTip("Review the threshold and start detection.")
        self.analyze_button.clicked.connect(self.show_detection_configuration)
        layout.addWidget(self.analyze_button)
        self.review_button = QPushButton("Observation review")
        self.review_button.setEnabled(False)
        self.review_button.clicked.connect(self.show_observation_review)
        layout.addWidget(self.review_button)
        self.open_review_button = QPushButton("Open saved review…")
        self.open_review_button.clicked.connect(self.open_saved_review)
        layout.addWidget(self.open_review_button)

        layout.addSpacing(13)
        layout.addWidget(self._section("ANALYSIS STATUS"))
        status_card = QFrame()
        status_card.setObjectName("statusCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 12, 12, 12)
        self.status_title = QLabel("Ready")
        self.status_title.setObjectName("statusTitle")
        self.status_detail = QLabel("")
        self.status_detail.setObjectName("mutedText")
        self.status_detail.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(5)
        self.progress.hide()
        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.status_detail)
        status_layout.addWidget(self.progress)
        layout.addWidget(status_card)
        layout.addStretch()

        model_heading = self._section("TRAINED MODEL")
        layout.addWidget(model_heading)
        self.model_label = QLabel("No checkpoint found")
        self.model_label.setTextFormat(Qt.TextFormat.PlainText)
        self.model_label.setObjectName("mutedText")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        return panel

    def _workspace(self):
        panel = QFrame()
        panel.setObjectName("workspace")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 15)
        layout.setSpacing(10)
        heading_row = QHBoxLayout()
        self.workspace_title = QLabel("VIDEO WEAPON DETECTION")
        self.workspace_title.setObjectName("workspaceTitle")
        self.workspace_meta = QLabel("Original evidence")
        self.workspace_meta.setObjectName("mutedText")
        heading_row.addWidget(self.workspace_title)
        heading_row.addStretch()
        heading_row.addWidget(self.workspace_meta)
        layout.addLayout(heading_row)

        enhancement = QFrame()
        enhancement.setObjectName("enhancementCard")
        enhancement.setToolTip("Open the automatic BasicVSR++ enhancement preview.")
        enhancement_layout = QHBoxLayout(enhancement)
        enhancement_layout.setContentsMargins(13, 10, 11, 10)
        enhancement_layout.setSpacing(12)
        enhancement_copy = QVBoxLayout()
        enhancement_copy.setSpacing(2)
        enhancement_heading = QHBoxLayout()
        enhancement_title = QLabel("VIDEO ENHANCEMENT · BASICVSR++")
        enhancement_title.setObjectName("enhancementTitle")
        enhancement_heading.addWidget(enhancement_title)
        enhancement_heading.addStretch()
        enhancement_copy.addLayout(enhancement_heading)
        self.enhancement_status = QLabel(
            "Import a video to preview automatic enhancement before detection"
        )
        self.enhancement_status.setObjectName("enhancementStatus")
        self.enhancement_status.setWordWrap(True)
        enhancement_copy.addWidget(self.enhancement_status)
        enhancement_layout.addLayout(enhancement_copy, 1)
        self.enhancement_button = QPushButton("Enhance video")
        self.enhancement_button.setObjectName("enhancementButton")
        self.enhancement_button.setToolTip("Open the BasicVSR++ enhancement preview.")
        self.enhancement_button.setEnabled(False)
        self.enhancement_button.clicked.connect(self.show_video_enhancement)
        enhancement_layout.addWidget(self.enhancement_button)
        layout.addWidget(enhancement)

        self.player = VideoPlayer()
        self.canvas = self.player.canvas
        self.canvas.browse_requested.connect(self.choose_video)
        self.canvas.file_dropped.connect(self.load_video)
        self.player.frame_changed.connect(self._frame_changed)
        self.player.failed.connect(lambda message: self._show_error("Playback unavailable", message))
        layout.addWidget(self.player, 1)
        observations = QFrame()
        observations.setObjectName("observationsPanel")
        observations_layout = QVBoxLayout(observations)
        observations_layout.setContentsMargins(10, 10, 10, 8)
        observations_layout.setSpacing(5)
        observations_layout.addWidget(self._section("DETECTED OBJECTS · CLICK A ROW TO REVIEW THE FRAME"))
        self.observation_model = DetectionTableModel(self)
        self.observations = QTableView()
        self.observations.setModel(self.observation_model)
        self.observations.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.observations.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.observations.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.observations.verticalHeader().hide()
        self.observations.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.observations.setMinimumHeight(130)
        self.observations.setMaximumHeight(170)
        self.observations.clicked.connect(self._jump_to_detection)
        observations_layout.addWidget(self.observations)
        layout.addWidget(observations)
        footer = QHBoxLayout()
        safety = QLabel("Automated detections require human review.")
        safety.setObjectName("safetyNote")
        self.dimensions_label = QLabel("—")
        self.dimensions_label.setObjectName("mutedText")
        footer.addWidget(safety)
        footer.addStretch()
        footer.addWidget(self.dimensions_label)
        layout.addLayout(footer)
        return panel

    def _summary_panel(self):
        panel = QFrame()
        panel.setObjectName("summaryPanel")
        panel.setMinimumWidth(275)
        panel.setMaximumWidth(390)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(17, 19, 17, 16)
        layout.setSpacing(10)
        layout.addWidget(self._section("DETECTION SUMMARY"))
        metrics = QHBoxLayout()
        total_card, self.total_metric = self._metric("—", "DETECTIONS")
        handgun_card, self.handgun_metric = self._metric("—", "HANDGUNS")
        knife_card, self.knife_metric = self._metric("—", "KNIVES")
        for card in (total_card, handgun_card, knife_card):
            metrics.addWidget(card)
        layout.addLayout(metrics)

        self.summary_message = QLabel("Import a video, review the confidence threshold, then start detection.")
        self.summary_message.setObjectName("summaryMessage")
        self.summary_message.setWordWrap(True)
        layout.addWidget(self.summary_message)
        layout.addWidget(self._section("CURRENT FRAME"))
        self.frame_message = QLabel("No video analyzed yet.")
        self.frame_message.setObjectName("mutedText")
        self.frame_message.setWordWrap(True)
        layout.addWidget(self.frame_message)

        scroll = QScrollArea()
        scroll.setObjectName("resultsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        scroll.setWidget(self.results_container)
        layout.addWidget(scroll, 1)
        self.run_meta = QLabel("Counts are observations across frames; a weapon may appear in several frames.")
        self.run_meta.setObjectName("mutedText")
        self.run_meta.setWordWrap(True)
        layout.addWidget(self.run_meta)
        return panel

    @staticmethod
    def _section(text: str):
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _metric(value: str, caption: str):
        card = QFrame()
        card.setObjectName("metricCard")
        box = QVBoxLayout(card)
        box.setContentsMargins(5, 8, 5, 8)
        box.setSpacing(1)
        number = QLabel(value)
        number.setObjectName("metricNumber")
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QLabel(caption)
        text.setObjectName("metricLabel")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(number)
        box.addWidget(text)
        return card, number

    def _apply_style(self):
        self.setStyleSheet((MOCKUP_DIR / "styles.qss").read_text(encoding="utf-8"))

    def choose_video(self):
        if self._thread:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import CCTV video", str(ROOT_DIR),
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.m4v)",
        )
        if filename:
            self.load_video(filename)

    def load_video(self, filename):
        if self._thread:
            return
        self._model_retry.stop()
        if self._config_dialog:
            self._config_dialog.reject()
        try:
            video = read_video(filename)
            self.player.open(video.path)
            self.player.set_locked(True)
        except (VideoInputError, OSError, ValueError) as exc:
            self._show_error("Video unavailable", str(exc))
            return
        self.video_info = video
        self.source_path = video.path
        self.result = None
        self._analysis_requested = False
        self.review_button.setEnabled(False)
        self.source_name.setText(video.path.name)
        self.source_name.setToolTip(str(video.path))
        self.source_meta.setText(f"{video.width} × {video.height} pixels\n{video.fps:.2f} fps · {timecode(video.duration)}")
        self.dimensions_label.setText(f"{video.frame_count:,} frames · Video only")
        self.original_button.setChecked(True)
        self.detected_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.analyze_button.setEnabled(True)
        self.enhancement_button.setEnabled(True)
        self.threshold_slider.setEnabled(True)
        self.workspace_meta.setText("Imported · ready to configure")
        self.enhancement_status.setText(
            "Video ready · Preview automatic BasicVSR++ enhancement before detection"
        )
        self._clear_results()
        self._update_model_label()
        self.status_title.setText("Ready to analyze")
        self.status_detail.setText(f"Selected threshold: {self.threshold_slider.value()}%. Review the settings to continue.")
        self.summary_message.setText("Video imported successfully. Detection has not started.")
        self.show_detection_configuration()

    def show_detection_configuration(self):
        if self.video_info is None or self._thread:
            return
        if self._config_dialog and self._config_dialog.isVisible():
            self._config_dialog.raise_()
            self._config_dialog.activateWindow()
            return
        dialog = DetectionConfigDialog(
            self.video_info, self.threshold_slider.value(), self,
            current_model=self.bridge.model_path,
        )
        self._config_dialog = dialog
        dialog.finished.connect(self._configuration_finished)
        dialog.open()

    def show_video_enhancement(self):
        if self.video_info is None or self._thread:
            return
        dialog = VideoEnhancementDialog(self.video_info, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.enhanced_video_path:
            try:
                new_info = read_video(dialog.enhanced_video_path)
                self.video_info = new_info
                self.player.load(self.video_info)
                self.enhancement_status.setText("BasicVSR++ Enhanced Video Active · Ready for detection")
                self.status_detail.setText(f"Enhanced video loaded: {new_info.path.name} · Selected threshold: {self.threshold_slider.value()}%")
                QMessageBox.information(
                    self,
                    "Enhanced Video Loaded",
                    f"Successfully applied BasicVSR++ super-resolution!\n\n"
                    f"Enhanced video: {new_info.path.name}\n"
                    f"Resolution: {new_info.width} × {new_info.height}\n\n"
                    "Click 'Analyze video' to run detection on the enhanced footage.",
                )
            except Exception as exc:
                QMessageBox.warning(self, "Enhanced Video Load Error", f"Could not load enhanced video:\n{exc}")

    @Slot(int)
    def _configuration_finished(self, code):
        dialog = self._config_dialog
        self._config_dialog = None
        if dialog is None or code != QDialog.DialogCode.Accepted:
            self.status_title.setText("Ready to analyze")
            self.status_detail.setText(f"Video remains imported · Selected threshold: {self.threshold_slider.value()}%")
            return
        self.threshold_slider.setValue(dialog.slider.value())
        if hasattr(dialog, "selected_model") and dialog.selected_model:
            self.bridge.set_model_path(dialog.selected_model)
        if hasattr(dialog, "cctv_intel_checkbox"):
            self.bridge.enable_cctv_intelligence = dialog.cctv_intel_checkbox.isChecked()
        if hasattr(dialog, "temporal_checkbox"):
            self.bridge.enable_temporal_consistency = dialog.temporal_checkbox.isChecked()
        self._update_model_label()
        self._analysis_requested = True
        self.analyze()

    def _update_model_label(self):
        # Completed/reopened results retain the checkpoint that produced them,
        # even if another model is selected or the weights have since been moved.
        path = Path(self.result.model_path) if self.result else self.bridge.model_path
        if path and (self.result is not None or path.is_file()):
            self.model_label.setText(f"Faster R-CNN · Handgun / Knife\n{path.name}")
            self.model_label.setToolTip(str(path))
        elif path:
            self.model_label.setText(f"Selected checkpoint unavailable\n{path.name}")
            self.model_label.setToolTip(str(path))
        else:
            self.model_label.setText("No checkpoint selected")
            self.model_label.setToolTip("Select an available .pth checkpoint in the detection configuration.")

    def analyze(self):
        if self.video_info is None or self._thread or not self._analysis_requested:
            return
        if not self.bridge.find_model():
            self._update_model_label()
            self.status_title.setText("Waiting for trained model")
            self.status_detail.setText(f"Place {MODEL_FILENAME} in mockup_ui/models. Analysis will start automatically when it is found.")
            self.summary_message.setText("Video imported. Waiting for the trained handgun/knife model before scanning.")
            self.workspace_meta.setText("Waiting for trained model")
            if not self._model_retry.isActive():
                self._model_retry.start()
            return
        self._model_retry.stop()
        self.player.pause()
        self.result = None
        self._update_model_label()
        self._clear_results()
        self.original_button.setChecked(True)
        try:
            self.player.open(self.video_info.path)
            self.player.set_locked(True)
        except ValueError as exc:
            self.detected_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self._show_error("Playback unavailable", str(exc))
            return
        for button in (self.open_button, self.save_button,
                       self.report_button, self.analyze_button, self.review_button, self.open_review_button,
                       self.original_button, self.detected_button, self.enhancement_button):
            button.setEnabled(False)
        self.threshold_slider.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status_title.setText("Loading detector")
        self.status_detail.setText("The detection service is loading the trained model to scan the entire video.")
        self.enhancement_status.setText(
            "Enhancement not applied · The detector is analyzing the original imported video"
        )
        self.workspace_meta.setText("Scanning · please wait")
        self.summary_message.setText("Analyzing the video for weapons. Detection totals will appear after processing completes.")
        self.frame_message.setText("Awaiting the completed detection result.")
        self._processing_dialog = ProcessingDialog(self.video_info.frame_count, self)
        self._processing_dialog.show()
        self._thread = InferenceWorker(self.bridge, self.video_info, self.threshold_slider.value() / 100, self)
        self._thread.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        self._thread.succeeded.connect(self._analysis_succeeded, Qt.ConnectionType.QueuedConnection)
        self._thread.failed.connect(self._analysis_failed, Qt.ConnectionType.QueuedConnection)
        self._thread.finished.connect(self._thread_finished, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    @Slot(str, int)
    def _on_progress(self, message, percent):
        self.status_detail.setText(message)
        if percent < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
            self.status_title.setText(f"Scanning video · {percent}%")
        if self._processing_dialog:
            self._processing_dialog.update_progress(message, percent)

    @Slot(object)
    def _analysis_succeeded(self, result):
        self.result = result
        self._update_model_label()
        if self._processing_dialog:
            self._processing_dialog.accept()
            self._processing_dialog = None
        try:
            ReviewStore.for_result(result)
        except (OSError, ValueError, KeyError, TypeError):
            logging.exception("Could not persist the completed analysis for review")
            self._show_error("Review storage unavailable", "The video completed, but the review record could not be stored. Check disk space and permissions before saving analyst decisions.")
        self._detections_by_frame = result.by_frame
        self._render_results(result)
        self.detected_button.setEnabled(True)
        self.detected_button.setChecked(True)
        self.report_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.review_button.setEnabled(bool(result.detections))
        if self._processing_dialog:
            self._processing_dialog.accept()
            self._processing_dialog = None
        try:
            self.player.open(result.output_path)
            self.player.set_locked(False)
            self.player.play()
            self.workspace_meta.setText("Annotated video")
        except ValueError as exc:
            self._show_error("Playback unavailable", str(exc))
        self.status_title.setText("Detection complete" if result.detections else "No weapons detected")
        self.status_detail.setText(
            f"{result.analyzed_frames:,} frames analyzed\n"
            f"{len(result.detections):,} frame detections · {result.elapsed_seconds:.1f} s"
        )
        self.enhancement_status.setText(
            "Enhancement not applied to this run · The original imported video was analyzed"
        )

    @Slot(str, str)
    def _analysis_failed(self, message, details):
        logging.error("Video inference failed\n%s", details)
        if self._processing_dialog:
            self._processing_dialog.accept()
            self._processing_dialog = None
        self._restore_after_incomplete()
        self.status_title.setText("Analysis unavailable")
        self.status_detail.setText(message)
        self.summary_message.setText("No completed detection result was produced.")
        self._show_error("Analysis unavailable", message)

    def _restore_after_incomplete(self):
        self.player.pause()
        self.result = None
        self._clear_results()
        self.save_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.review_button.setEnabled(False)
        self.detected_button.setEnabled(False)
        self.original_button.setChecked(True)
        self.show_original()
        self.player.set_locked(True)

    @Slot()
    def _thread_finished(self):
        # The built-in QThread.finished signal fires after run() returns. Join
        # before disposal, keeping both Qt and Python ownership on the UI thread.
        completed_thread = self._thread
        if completed_thread is None:
            return
        completed_thread.wait()
        completed_thread.deleteLater()
        self.progress.hide()
        self.open_button.setEnabled(True)
        self.open_review_button.setEnabled(True)
        self.analyze_button.setEnabled(self.video_info is not None)
        self.enhancement_button.setEnabled(self.video_info is not None)
        self.report_button.setEnabled(self.result is not None)
        self.original_button.setEnabled(True)
        self.threshold_slider.setEnabled(True)
        self._thread = None
        self._update_model_label()

    def _render_results(self, result):
        self.observation_model.set_records(result.detections)
        self.total_metric.setText(f"{len(result.detections):,}")
        self.handgun_metric.setText(f"{result.counts.get('handgun', 0):,}")
        self.knife_metric.setText(f"{result.counts.get('knife', 0):,}")
        if result.detections:
            self.summary_message.setText(
                f"Weapons detected in {result.positive_frames:,} of {result.analyzed_frames:,} frames. "
                "Use Observation review to record Accept, Reject, or Uncertain for each detection."
            )
        else:
            self.summary_message.setText(
                f"No handgun or knife met the {result.threshold:.0%} confidence threshold in this video."
            )
        self.run_meta.setText(
            f"Threshold {result.threshold:.0%} · {result.device.upper()} · Every frame\n"
            "Counts are frame observations; a weapon may appear in several frames."
        )

    def _frame_changed(self, number):
        self._clear_result_cards()
        if self.result is None:
            self.frame_message.setText("The annotated video and detection summary appear after automatic analysis.")
            return
        rows = self._detections_by_frame.get(number, [])
        self.frame_message.setText(
            f"{timecode(number / self.video_info.fps)} · Frame {number}\n"
            + (f"{len(rows)} detection(s) in this frame" if rows else "No detections in this frame")
        )
        for index, row in enumerate(rows, 1):
            self.results_layout.insertWidget(index - 1, DetectionCard(row, index))

    def _jump_to_detection(self, index):
        if self.result is None or not index.isValid() or self._thread:
            return
        row = self.observation_model.records[index.row()]
        self.detected_button.setChecked(True)
        self._switch_view(self.result.output_path, row["frame_number"], "Annotated video")

    def _switch_view(self, path, frame_number, label):
        try:
            if self.player.path != Path(path).resolve():
                self.player.open(path, frame_number)
            else:
                self.player.seek_frame(frame_number)
            self.workspace_meta.setText(label)
        except ValueError as exc:
            self._show_error("Playback unavailable", str(exc))

    def show_original(self):
        if self.video_info is not None:
            self._switch_view(self.video_info.path, self.player.frame_number, "Original CCTV footage")

    def show_detected(self):
        if self.result is not None:
            self._switch_view(self.result.output_path, self.player.frame_number, "Annotated video")

    def _clear_result_cards(self):
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _clear_results(self):
        self._detections_by_frame = {}
        self._clear_result_cards()
        self.observation_model.set_records([])
        for metric in (self.total_metric, self.handgun_metric, self.knife_metric):
            metric.setText("—")
        self.frame_message.setText("No video analyzed yet.")
        self.summary_message.setText("Import a video, review the confidence threshold, then start detection.")
        self.run_meta.setText("Counts are observations across frames; a weapon may appear in several frames.")

    def save_current_result(self):
        if not self.result:
            return
        destination = QFileDialog.getExistingDirectory(
            self, "Choose where to save the video and forensic report", str(Path.home())
        )
        if not destination:
            return
        try:
            directory = save_result(self.result, Path(destination))
        except Exception:
            logging.exception("Could not save video result")
            self._show_error("Save failed", "The video/report could not be saved. Check available disk space and permissions for the selected destination.")
            return
        QMessageBox.information(self, "Video and report saved",
                                "Saved annotated video, original detection records, and a forensic report.\n"
                                "The PDF includes saved analyst reviews. Reopen observation_reviews.json to continue reviewing. Structured records are in "
                                f"forensic_report.json and forensic_records.csv.\n\n{directory}")

    def show_forensic_report(self):
        if self.result is None or self._thread:
            return
        try:
            ForensicReportDialog(self.result, self).exec()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._show_error("Report unavailable", str(exc))

    def show_observation_review(self):
        if self.result is None or self._thread or not self.result.detections:
            return
        self.player.pause()
        try:
            ObservationReviewDialog(self.result, self).exec()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._show_error("Observation review unavailable", str(exc))

    def open_saved_review(self):
        if self._thread:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Open a saved observation review", str(REVIEW_DIR), "Observation review (*.json)")
        if not filename:
            return
        try:
            result = ReviewStore(filename).restore_result()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._show_error("Review file unavailable", f"This file could not be opened as an observation review.\n{exc}")
            return
        self._model_retry.stop()
        if self._config_dialog:
            self._config_dialog.reject()
        self._analysis_requested = False
        self.video_info, self.source_path = result.video, result.video.path
        self.source_name.setText(result.video.path.name)
        self.source_name.setToolTip(str(result.video.path))
        self.source_meta.setText(f"{result.video.width} × {result.video.height} pixels\n{result.video.fps:.2f} fps · {timecode(result.video.duration)}")
        self.threshold_slider.setValue(round(result.threshold * 100))
        self.analyze_button.setEnabled(result.video.path.is_file())
        self.enhancement_button.setEnabled(result.video.path.is_file())
        self._analysis_succeeded(result)
        self.enhancement_status.setText(
            "Enhancement not applied to this saved run · The original imported video was analyzed"
        )
        self.player.pause()
        self.show_observation_review()

    def _set_ready(self, detail):
        if not self.bridge.model_path or not self.bridge.model_path.is_file():
            self.status_title.setText("Automatic model lookup")
            self.status_detail.setText(f"Place {MODEL_FILENAME} in mockup_ui/models, then import a video.")
        else:
            self.status_title.setText("Ready for detection")
            self.status_detail.setText(detail)

    def _show_error(self, title: str, message: str):
        QMessageBox.warning(self, title, message)

    def closeEvent(self, event):
        if self._thread:
            QMessageBox.information(self, "Video analysis in progress", "Let the current video analysis finish before closing the window.")
            event.ignore()
            return
        self.player.release()
        self._model_retry.stop()
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description="Forensikada forensic video weapon detection")
    parser.add_argument("--video", help="Optional CCTV video to analyze automatically on startup")
    return parser.parse_args()


def main():
    args = parse_args()
    application = QApplication(sys.argv[:1])
    application.setApplicationName("Forensikada Video Analysis")
    application.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f5f6f8"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2f318e"))
    application.setPalette(palette)
    window = MainWindow(initial_video=args.video)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
