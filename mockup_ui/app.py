"""Forensikada defense mockup: video recognition and weapon detection UI."""

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
    QFileDialog,
    QFrame,
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
)

from mockup_ui.model_bridge import (
    MODEL_FILENAME, ModelBridge, ModelSetupError, VideoAnalysisResult, VideoInfo, VideoInputError,
    read_video, save_result, timecode,
)
from mockup_ui.video_player import VideoPlayer

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
        sub = QLabel("CCTV FORENSIC ANALYSIS · DEFENSE MOCKUP")
        sub.setObjectName("brandSub")
        brand.addWidget(title)
        brand.addWidget(sub)
        layout.addLayout(brand)
        layout.addStretch()

        self.open_button = QPushButton("Import video")
        self.open_button.setObjectName("primaryButton")
        self.open_button.setToolTip("Import a video to start weapon analysis automatically.")
        self.open_button.setIcon(QIcon(str(MOCKUP_DIR / "assets" / "import-image.png")))
        self.open_button.clicked.connect(self.choose_video)
        self.save_button = QPushButton("Save video + report")
        self.save_button.setToolTip("Save the annotated video, forensic report, and structured detection records.")
        self.save_button.setObjectName("headerButton")
        self.save_button.setIcon(QIcon(str(MOCKUP_DIR / "assets" / "save-result.svg")))
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current_result)
        for button in (self.open_button, self.save_button):
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

        self.summary_message = QLabel("Import a video to automatically analyze it for handguns and knives.")
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
        self.source_name.setText(video.path.name)
        self.source_meta.setText(f"{video.width} × {video.height} pixels\n{video.fps:.2f} fps · {timecode(video.duration)}")
        self.dimensions_label.setText(f"{video.frame_count:,} frames · Video only")
        self.original_button.setChecked(True)
        self.detected_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.workspace_meta.setText("Imported · awaiting analysis")
        self._clear_results()
        self._update_model_label()
        self.analyze()

    def _update_model_label(self):
        self.bridge.find_model()
        path = self.bridge.model_path
        available = bool(path and path.is_file())
        if available:
            self.model_label.setText(f"Faster R-CNN · Handgun / Knife\n{path.name}")
            self.model_label.setToolTip(str(path))
        else:
            self.model_label.setText(f"Trained model not found\n{MODEL_FILENAME}")
            self.model_label.setToolTip(f"Place {MODEL_FILENAME} in mockup_ui/models.")

    def analyze(self):
        if self.video_info is None or self._thread:
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
        self._update_model_label()
        self.player.pause()
        self.result = None
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
                       self.original_button, self.detected_button):
            button.setEnabled(False)
        self.threshold_slider.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status_title.setText("Loading detector")
        self.status_detail.setText("The detection service is loading the trained model to scan the entire video.")
        self.workspace_meta.setText("Scanning · please wait")
        self.summary_message.setText("Analyzing the video for weapons. Detection totals will appear after processing completes.")
        self.frame_message.setText("Awaiting the completed detection result.")
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

    @Slot(object)
    def _analysis_succeeded(self, result):
        self.result = result
        self._detections_by_frame = result.by_frame
        self._render_results(result)
        self.detected_button.setEnabled(True)
        self.detected_button.setChecked(True)
        self.save_button.setEnabled(True)
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

    @Slot(str, str)
    def _analysis_failed(self, message, details):
        logging.error("Video inference failed\n%s", details)
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
                "Play the result or click an observation to review its bounding box."
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
        self.summary_message.setText("Import a video to automatically analyze it for handguns and knives.")
        self.run_meta.setText("Counts are observations across frames; a weapon may appear in several frames.")

    def save_current_result(self):
        if not self.result:
            return
        try:
            directory = save_result(self.result)
        except Exception:
            logging.exception("Could not save video result")
            self._show_error("Save failed", "The video/report could not be saved. Check disk space in mockup_ui/outputs.")
            return
        QMessageBox.information(self, "Video and report saved",
                                "Saved annotated video, original detection records, and a forensic report.\n"
                                "Open forensic_report.html to read or print it. Structured records are in "
                                f"forensic_report.json and forensic_records.csv.\n\n{directory}")

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
    parser = argparse.ArgumentParser(description="Forensikada video weapon detection defense mockup")
    parser.add_argument("--video", help="Optional CCTV video to analyze automatically on startup")
    return parser.parse_args()


def main():
    args = parse_args()
    application = QApplication(sys.argv[:1])
    application.setApplicationName("Forensikada Defense Mockup")
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
