import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def main():
    application = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(application.exec())


if __name__ == "__main__":
    main()