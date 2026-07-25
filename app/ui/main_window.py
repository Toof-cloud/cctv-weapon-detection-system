from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CCTV Weapon Detection System")
        self.resize(1100, 700)

        self.create_interface()

    def create_interface(self):
        container = QWidget()
        main_layout = QVBoxLayout(container)

        title = QLabel("CCTV Weapon Detection System")
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold;"
        )

        description = QLabel(
            "Upload and analyze a 20 to 30-second "
            "prerecorded CCTV video."
        )

        camera_layout = QHBoxLayout()

        camera_label = QLabel("Camera ID:")

        self.camera_id_input = QLineEdit()
        self.camera_id_input.setPlaceholderText(
            "Example: CAM-01"
        )

        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_id_input)

        self.select_video_button = QPushButton(
            "Select CCTV Video"
        )

        self.selected_file_label = QLabel(
            "No video selected"
        )

        metadata_layout = QGridLayout()

        metadata_layout.addWidget(
            QLabel("Duration:"), 0, 0
        )
        self.duration_value = QLabel("N/A")
        metadata_layout.addWidget(
            self.duration_value, 0, 1
        )

        metadata_layout.addWidget(
            QLabel("Resolution:"), 1, 0
        )
        self.resolution_value = QLabel("N/A")
        metadata_layout.addWidget(
            self.resolution_value, 1, 1
        )

        metadata_layout.addWidget(
            QLabel("Frame rate:"), 2, 0
        )
        self.fps_value = QLabel("N/A")
        metadata_layout.addWidget(
            self.fps_value, 2, 1
        )

        metadata_layout.addWidget(
            QLabel("Total frames:"), 3, 0
        )
        self.frame_count_value = QLabel("N/A")
        metadata_layout.addWidget(
            self.frame_count_value, 3, 1
        )

        self.analyze_button = QPushButton(
            "Analyze Video"
        )
        self.analyze_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        results_label = QLabel("Detection Results")
        results_label.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Timestamp",
                "Frame",
                "Weapon",
                "Confidence",
            ]
        )

        self.export_button = QPushButton(
            "Export CSV Report"
        )
        self.export_button.setEnabled(False)

        self.status_label = QLabel("Status: Ready")

        main_layout.addWidget(title)
        main_layout.addWidget(description)
        main_layout.addLayout(camera_layout)
        main_layout.addWidget(
            self.select_video_button
        )
        main_layout.addWidget(
            self.selected_file_label
        )
        main_layout.addLayout(metadata_layout)
        main_layout.addWidget(self.analyze_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(results_label)
        main_layout.addWidget(self.results_table)
        main_layout.addWidget(self.export_button)
        main_layout.addWidget(self.status_label)

        self.setCentralWidget(container)