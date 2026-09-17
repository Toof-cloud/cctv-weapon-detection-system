"""Multi-recording workspace built on the same full-video detector as the single view."""
from pathlib import Path
from time import monotonic
import traceback

from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QDialogButtonBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QSizePolicy)

from mockup_ui.model_bridge import ModelBridge, read_video, timecode
from mockup_ui.multi_camera import Alignment, CameraSource, build_timeline, export_session, validate_session
from mockup_ui.observation_review import ReviewStore
from mockup_ui.review_panel import ObservationReviewDialog
from mockup_ui.video_player import VideoPlayer

VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.m4v)"


def label(text, name="dialogDetail"):
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    return widget


def session_time(seconds):
    return ("-" if seconds < 0 else "") + timecode(abs(seconds))


class CameraBatchWorker(QThread):
    progress = Signal(str, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, sources, bridge, threshold, parent=None):
        super().__init__(parent)
        self.sources, self.bridge, self.threshold = sources, bridge, threshold

    def run(self):
        try:
            results = []
            for index, source in enumerate(self.sources):
                def progress(message, percent):
                    self.progress.emit(f"{source.camera_id} · {index + 1}/{len(self.sources)} cameras · {message}",
                                       round((index + max(0, percent) / 100) * 100 / len(self.sources)))
                result = self.bridge.analyze_video(source.video, self.threshold, progress, camera_id=source.camera_id)
                ReviewStore.for_result(result)
                results.append(result)
            self.succeeded.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


class CameraSetupDialog(QDialog):
    """Keeps technical correspondence fields available without crowding the workspace."""
    def __init__(self, sources, alignment, parent=None):
        super().__init__(parent)
        self.sources = sources
        self.alignment = alignment
        self.editors = []
        self.setWindowTitle("Camera and timing details")
        self.setObjectName("cameraSetupDialog")
        self.resize(620, 650)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.addWidget(label("CAMERA & TIMING DETAILS", "sectionTitle"))
        layout.addWidget(label("Use offsets only when recordings did not begin at the same incident time.", "dialogHeading"))
        layout.addWidget(label("Session time = video time + offset. Camera IDs must be unique."))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        stack = QVBoxLayout(content)
        for source in sources:
            card = QFrame()
            card.setObjectName("configCard")
            box = QVBoxLayout(card)
            box.addWidget(label(source.video.path.name, "detectionName"))
            identifier = QLineEdit(source.camera_id)
            identifier.setPlaceholderText("Camera ID")
            location = QLineEdit(source.location)
            location.setPlaceholderText("Viewpoint / location (optional)")
            offset = QDoubleSpinBox()
            offset.setRange(-86400, 86400)
            offset.setDecimals(3)
            offset.setSuffix(" s")
            offset.setValue(source.offset_seconds)
            offset.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            box.addWidget(label("Camera ID"))
            box.addWidget(identifier)
            box.addWidget(label("Viewpoint / location"))
            box.addWidget(location)
            timing = QHBoxLayout()
            timing.addWidget(label("Start offset"))
            timing.addWidget(offset)
            box.addLayout(timing)
            stack.addWidget(card)
            self.editors.append((identifier, location, offset))
        stack.addWidget(label("ALIGNMENT OPTIONS", "sectionTitle"))
        self.method = QLineEdit(alignment.method)
        self.method.setPlaceholderText("e.g. shared clock or visible timer at 00:12")
        stack.addWidget(label("Alignment reference"))
        stack.addWidget(self.method)
        row = QHBoxLayout()
        row.addWidget(label("Matching tolerance"))
        self.window = QDoubleSpinBox()
        self.window.setRange(.05, 60)
        self.window.setDecimals(2)
        self.window.setSuffix(" s")
        self.window.setValue(alignment.window_seconds)
        self.window.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        row.addWidget(self.window)
        stack.addLayout(row)
        stack.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def validate_and_accept(self):
        identifiers = [editor[0].text().strip() for editor in self.editors]
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            QMessageBox.warning(self, "Check camera IDs", "Each camera needs a different, non-empty ID.")
            return
        self.accept()

    def apply(self):
        for source, (identifier, location, offset) in zip(self.sources, self.editors, strict=True):
            source.camera_id = identifier.text().strip()
            source.location = location.text().strip()
            source.offset_seconds = offset.value()
        self.alignment.method = self.method.text().strip()
        self.alignment.window_seconds = self.window.value()


class CameraPreviewDialog(QDialog):
    def __init__(self, path, title, frame_number=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Camera preview · {title}")
        self.setObjectName("cameraPreviewDialog")
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.resize(1120, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.addWidget(label(title, "dialogHeading"))
        self.player = VideoPlayer()
        self.player.open(path, frame_number)
        self.player.start_button.hide()
        self.player.canvas.setToolTip("Scroll to zoom, drag to move, and double-click to fit the video.")
        layout.addWidget(self.player, 1)
        hint = label("Scroll to zoom · Drag to move · Double-click to fit")
        hint.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(hint)

    def closeEvent(self, event):
        self.player.release()
        event.accept()


class MultiCameraDialog(QDialog):
    def __init__(self, videos, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Forensikada · Multi-camera workspace")
        self.setObjectName("multiCameraDialog")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setMinimumSize(1060, 720)
        self.resize(1380, 900)
        self.setStyleSheet((Path(__file__).parent / "styles.qss").read_text(encoding="utf-8"))
        self.sources = [CameraSource(video, f"CAM-{index + 1:02d}") for index, video in enumerate(videos)]
        self.results, self.rows, self.cards, self.editors = [], [], [], []
        self.alignment = Alignment()
        self.bridge = ModelBridge()
        if parent and hasattr(parent, "bridge"):
            self.bridge.set_model_path(parent.bridge.model_path)
        self.threshold = 50
        self.worker = None
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self._tick)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("MULTI-CAMERA / POST-INCIDENT REVIEW", "sectionTitle"))
        titles.addWidget(label("One incident. Multiple viewpoints.", "dialogHeading"))
        header.addLayout(titles, 1)
        self.configure_button = QPushButton("Analyze cameras")
        self.configure_button.setObjectName("primaryButton")
        self.configure_button.clicked.connect(self.configure)
        self.export_button = QPushButton("Save session")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        header.addWidget(self.configure_button)
        header.addWidget(self.export_button)
        layout.addLayout(header)
        self.status = label("Name the incident, confirm whether the recordings match, then analyze. Original files are preserved.")
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        self.setup = QWidget()
        self.setup.setMinimumWidth(250)
        self.setup.setMaximumWidth(300)
        side = QVBoxLayout(self.setup)
        side.setContentsMargins(0, 0, 10, 0)
        side.addWidget(label("SESSION SETUP", "sectionTitle"))
        side.addWidget(label("Incident name or ID"))
        self.scene = QLineEdit()
        self.scene.setPlaceholderText("e.g. Scene 004")
        side.addWidget(self.scene)
        self.confirmed = QCheckBox("Same incident and timing aligned")
        self.confirmed.setToolTip("Confirm after checking that the recordings correspond and their timing is aligned.")
        side.addWidget(self.confirmed)
        side.addWidget(label("If the videos did not start together, adjust their timing before analysis."))
        self.details_button = QPushButton("Camera and timing details")
        self.details_button.clicked.connect(self.open_details)
        side.addWidget(self.details_button)
        add = QPushButton("+ Add camera recordings")
        add.clicked.connect(self.add_recordings)
        side.addWidget(add)
        self.source_scroll = QScrollArea()
        self.source_scroll.setWidgetResizable(True)
        side.addWidget(self.source_scroll, 1)
        side.addWidget(label("Cross-camera matches support review; they do not prove that two views show the same physical object.", "enhancementNotice"))
        splitter.addWidget(self.setup)

        content = QWidget()
        right = QVBoxLayout(content)
        right.setContentsMargins(8, 0, 0, 0)
        tools = QHBoxLayout()
        self.camera_count = label("CAMERA VIEWS", "sectionTitle")
        tools.addWidget(self.camera_count, 1)
        self.view = QComboBox()
        self.view.addItems(["Imported recordings", "Annotated results"])
        self.view.setEnabled(False)
        self.view.currentIndexChanged.connect(self.change_view)
        tools.addWidget(self.view)
        right.addLayout(tools)
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        right.addWidget(self.grid_scroll, 3)
        playback = QHBoxLayout()
        self.play_button = QPushButton("Play all")
        self.play_button.clicked.connect(self.toggle_playback)
        playback.addWidget(self.play_button)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.sliderPressed.connect(self.pause)
        self.seek.valueChanged.connect(self.seek_all)
        playback.addWidget(self.seek, 1)
        self.time_label = label("Session 00:00.000")
        playback.addWidget(self.time_label)
        right.addLayout(playback)
        self.metrics_label = label("COMBINED OBSERVATIONS · Awaiting analysis", "sectionTitle")
        right.addWidget(self.metrics_label)
        self.metric_note = label("Uncertain and Not Applicable are reported separately and excluded from MCCR.")
        right.addWidget(self.metric_note)
        self.timeline = QTableWidget(0, 8)
        self.timeline.setHorizontalHeaderLabels(["Session time", "Camera", "Video time", "Frame", "Object", "Confidence", "Cross-view status", "Analyst"])
        self.timeline.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.timeline.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.timeline.setAlternatingRowColors(True)
        self.timeline.verticalHeader().hide()
        self.timeline.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.timeline.horizontalHeader().setStretchLastSection(True)
        self.timeline.cellClicked.connect(self.jump_to_observation)
        self.timeline.setMinimumHeight(160)
        right.addWidget(self.timeline, 2)
        splitter.addWidget(content)
        splitter.setSizes([275, 1100])
        splitter.setStretchFactor(1, 1)
        self.rebuild_sources()

    def capture_settings(self):
        self.alignment.scene_id = self.scene.text().strip()
        self.alignment.confirmed = self.confirmed.isChecked()
        if self.alignment.confirmed and not self.alignment.method:
            self.alignment.method = "User-confirmed synchronized recordings"

    def open_details(self):
        self.capture_settings()
        dialog = CameraSetupDialog(self.sources, self.alignment, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.apply()
            self.rebuild_sources()
            self.status.setText("Camera and timing details saved. Confirm the incident, then analyze all cameras.")

    def rebuild_sources(self):
        self.pause()
        for card in self.cards:
            card[0].release()
        self.cards, self.editors = [], []
        source_widget, grid_widget = QWidget(), QWidget()
        source_layout, grid = QVBoxLayout(source_widget), QGridLayout(grid_widget)
        source_layout.setContentsMargins(0, 0, 5, 0)
        grid.setContentsMargins(0, 0, 0, 0)
        for index, source in enumerate(self.sources):
            editor = QFrame()
            editor.setObjectName("cameraEditor")
            box = QVBoxLayout(editor)
            box.addWidget(label(source.camera_id, "detectionName"))
            filename = label(source.video.path.name)
            filename.setToolTip(str(source.video.path))
            box.addWidget(filename)
            detail = source.location or "Location not specified"
            if source.offset_seconds:
                detail += f" · offset {source.offset_seconds:+.3f} s"
            box.addWidget(label(detail))
            remove = QPushButton("Remove")
            remove.clicked.connect(lambda checked=False, i=index: self.remove_recording(i))
            box.addWidget(remove)
            source_layout.addWidget(editor)
            card = QFrame()
            card.setObjectName("cameraCard")
            body = QVBoxLayout(card)
            body.setContentsMargins(10, 10, 10, 10)
            body.setSpacing(6)
            top = QHBoxLayout()
            title = label(source.camera_id, "enhancementModel")
            top.addWidget(title, 1)
            review = QPushButton("Review")
            review.setEnabled(False)
            review.clicked.connect(lambda checked=False, i=index: self.review_camera(i))
            preview = QPushButton("Preview")
            preview.clicked.connect(lambda checked=False, i=index: self.preview_camera(i))
            top.addWidget(preview)
            top.addWidget(review)
            body.addLayout(top)
            name = label(source.video.path.name)
            name.setToolTip(str(source.video.path))
            body.addWidget(name)
            player = VideoPlayer()
            player.canvas.setMinimumSize(260, 145)
            player.canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            player.canvas.setAcceptDrops(False)
            player.layout().itemAt(1).widget().hide()
            player.open(source.video.path)
            player.failed.connect(self.playback_error)
            body.addWidget(player, 1)
            info = label(f"{source.video.width} × {source.video.height} · {source.video.fps:.2f} FPS · {timecode(source.video.duration)}")
            body.addWidget(info)
            state = label("Imported · Ready for analysis", "cameraStatus")
            body.addWidget(state)
            grid.addWidget(card, index // 2, index % 2)
            self.cards.append((player, title, state, review))
        source_layout.addStretch()
        for scroll, widget in ((self.source_scroll, source_widget), (self.grid_scroll, grid_widget)):
            old = scroll.takeWidget()
            if old:
                old.deleteLater()
            scroll.setWidget(widget)
        self.camera_count.setText(f"CAMERA VIEWS / {len(self.sources):02d}")
        self.configure_button.setEnabled(len(self.sources) >= 2)
        self.update_range()

    def update_range(self):
        if self.sources:
            self.seek.setRange(round(min(s.offset_seconds for s in self.sources) * 1000),
                               round(max(s.offset_seconds + s.video.duration for s in self.sources) * 1000) - 1)
            self.seek.setValue(self.seek.minimum())
            self.seek_all(self.seek.value())

    def clear_results(self):
        self.results, self.rows = [], []
        self.timeline.setRowCount(0)
        self.export_button.setEnabled(False)
        self.view.setEnabled(False)
        self.view.setCurrentIndex(0)
        self.metrics_label.setText("COMBINED OBSERVATIONS · Awaiting analysis")

    def add_recordings(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Add camera recordings", "", VIDEO_FILTER)
        if not filenames:
            return
        self.capture_settings()
        existing = {s.video.path.resolve() for s in self.sources}
        try:
            videos = [read_video(path) for path in dict.fromkeys(filenames) if Path(path).resolve() not in existing]
        except Exception as exc:
            QMessageBox.warning(self, "Recording unavailable", str(exc))
            return
        for video in videos:
            number = 1
            while f"CAM-{number:02d}" in {s.camera_id for s in self.sources}:
                number += 1
            self.sources.append(CameraSource(video, f"CAM-{number:02d}"))
        self.clear_results()
        self.rebuild_sources()

    def remove_recording(self, index):
        self.capture_settings()
        self.sources.pop(index)
        self.clear_results()
        self.rebuild_sources()

    def configure(self):
        if self.worker:
            return
        if self.results:
            self.clear_results()
            self.rebuild_sources()
            self.setup.setEnabled(True)
            self.configure_button.setText("Analyze cameras")
            self.status.setText("Edit camera setup, then configure the next analysis. Previous exports remain unchanged.")
            return
        self.pause()
        self.capture_settings()
        try:
            validate_session(self.sources, self.alignment)
        except ValueError as exc:
            QMessageBox.warning(self, "Check camera setup", str(exc))
            return
        from mockup_ui.app import DetectionConfigDialog
        dialog = DetectionConfigDialog(self.sources[0].video, self.threshold, self, self.bridge.model_path,
                                       videos=[s.video for s in self.sources])
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.threshold = dialog.slider.value()
        self.bridge.set_model_path(dialog.selected_model)
        self.bridge.enable_cctv_intelligence = dialog.cctv_intel_checkbox.isChecked()
        self.bridge.enable_temporal_consistency = dialog.temporal_checkbox.isChecked()
        self.start_analysis()

    def start_analysis(self):
        self.clear_results()
        self.rebuild_sources()
        self.setup.setEnabled(False)
        self.configure_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.seek.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.worker = CameraBatchWorker(list(self.sources), self.bridge, self.threshold / 100, self)
        self.worker.progress.connect(self.on_progress, Qt.ConnectionType.QueuedConnection)
        self.worker.succeeded.connect(self.completed, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self.failed, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.worker_finished, Qt.ConnectionType.QueuedConnection)
        self.worker.start()

    @Slot(str, int)
    def on_progress(self, message, percent):
        self.status.setText(message)
        self.progress_bar.setValue(percent)

    @Slot(object)
    def completed(self, results):
        self.results = results
        self.rows, metrics = build_timeline(self.sources, results, self.alignment)
        counts = metrics["counts"]
        score = f"{metrics['mccr_percent']:.2f}%" if metrics["mccr_percent"] is not None else "N/A"
        self.metrics_label.setText(f"{len(self.rows)} observations  ·  {counts.get('Corroborated', 0)} corroborated  ·  MCCR {score}")
        self.metric_note.setText(f"{counts.get('Not Corroborated', 0)} not corroborated · {counts.get('Uncertain', 0)} uncertain · {counts.get('Not Applicable', 0)} not applicable. Uncertain and Not Applicable are excluded from MCCR.")
        self.refresh_timeline()
        for result, card in zip(results, self.cards, strict=True):
            card[2].setText(f"Analyzed · {len(result.detections)} observations · {result.analyzed_frames} frames")
            card[3].setEnabled(bool(result.detections))
        self.export_button.setEnabled(True)
        self.view.setEnabled(True)
        self.view.setCurrentIndex(1)
        self.status.setText("Analysis complete. Preview a camera, review detections, or select a row to inspect the matching time.")
        self.configure_button.setText("New analysis")

    @Slot(str)
    def failed(self, details):
        import logging
        logging.error("Multi-camera analysis failed\n%s", details)
        self.status.setText("Session analysis did not complete. Correct the input or model issue and retry.")
        QMessageBox.warning(self, "Camera analysis failed", details.splitlines()[-1])

    @Slot()
    def worker_finished(self):
        self.worker.wait()
        self.worker.deleteLater()
        self.worker = None
        self.setup.setEnabled(not bool(self.results))
        self.configure_button.setEnabled(len(self.sources) >= 2)
        self.play_button.setEnabled(True)
        self.seek.setEnabled(True)
        self.progress_bar.hide()
        if self.results:
            self.toggle_playback()

    def refresh_timeline(self):
        decisions = {}
        for result in self.results:
            decisions.update({row["observationId"]: (row["analystReview"] or {}).get("decision", "Not reviewed")
                              for row in ReviewStore.for_result(result).observations})
        self.timeline.setRowCount(len(self.rows))
        for index, row in enumerate(self.rows):
            values = [session_time(row["session_seconds"]), row["camera_id"], timecode(row["video_seconds"]),
                      str(row["frame_number"]), row["object_label"].title(), f"{row['confidence']:.1%}",
                      row["corroboration_status"], decisions.get(row["observation_id"], "Not reviewed")]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(f"{row['source_video']}\n{row['reason']}\nTemporal status: {row['temporal_status']}\nBox: {row['box']}")
                self.timeline.setItem(index, column, item)

    def review_camera(self, index):
        self.pause()
        _, metrics = build_timeline(self.sources, self.results, self.alignment)
        ObservationReviewDialog(self.results[index], self,
                                cross_view={row["observation_id"]: row for row in self.rows},
                                session_mccr=metrics["mccr_percent"]).exec()
        self.refresh_timeline()

    def preview_camera(self, index):
        self.pause()
        source = self.sources[index]
        path = self.results[index].output_path if self.results and self.view.currentIndex() == 1 else source.video.path
        frame = self.cards[index][0].frame_number
        CameraPreviewDialog(path, f"{source.camera_id} · {source.video.path.name}", frame, self).exec()

    def jump_to_observation(self, row, column):
        self.pause()
        self.seek.setValue(round(self.rows[row]["session_seconds"] * 1000))

    def change_view(self):
        self.pause()
        for index, (source, card) in enumerate(zip(self.sources, self.cards)):
            path = self.results[index].output_path if self.results and self.view.currentIndex() == 1 else source.video.path
            try:
                card[0].open(path)
            except ValueError as exc:
                self.playback_error(str(exc))
        self.seek_all(self.seek.value())

    def seek_all(self, milliseconds):
        seconds = milliseconds / 1000
        self.time_label.setText(f"Session {session_time(seconds)}")
        for source, card in zip(self.sources, self.cards):
            local = seconds - source.offset_seconds
            if 0 <= local < source.video.duration:
                card[0].seek_frame(min(round(local * source.video.fps), source.video.frame_count - 1))
                card[0].canvas.show()
            else:
                card[0].canvas._source_pixmap = QPixmap()
                card[0].canvas.clear()
                card[0].canvas.setText("No recording at this session time")
            card[1].setText(f"{source.camera_id} · {timecode(local) if 0 <= local < source.video.duration else 'Outside recording'}")

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.pause()
        elif self.sources:
            if self.seek.value() >= self.seek.maximum():
                self.seek.setValue(self.seek.minimum())
            self._play_start, self._play_position = monotonic(), self.seek.value()
            self.play_timer.start()
            self.play_button.setText("Pause all")

    def _tick(self):
        position = self._play_position + round((monotonic() - self._play_start) * 1000)
        self.seek.setValue(min(position, self.seek.maximum()))
        if position >= self.seek.maximum():
            self.pause()

    def pause(self):
        self.play_timer.stop()
        self.play_button.setText("Play all")

    def playback_error(self, message):
        self.pause()
        self.status.setText(message)

    def export(self):
        self.pause()
        destination = QFileDialog.getExistingDirectory(self, "Save camera session")
        if not destination:
            return
        try:
            folder = export_session(self.sources, self.results, self.alignment, destination)
        except Exception as exc:
            QMessageBox.warning(self, "Export incomplete", f"The session could not be fully saved: {exc}")
            return
        QMessageBox.information(self, "Session saved", f"Saved combined observations, alignment metadata, and a video/report/review package for each camera.\n\n{folder}")

    def reject(self):
        self.close()

    def accept(self):
        self.close()

    def closeEvent(self, event):
        if self.worker:
            self.status.setText("The detector is processing recordings. Wait for completion before closing this workspace.")
            event.ignore()
            return
        self.pause()
        for player, *_ in self.cards:
            player.release()
        event.accept()
