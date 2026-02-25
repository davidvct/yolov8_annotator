"""
YOLOv8 Annotator - A GUI application for annotating images with polygon annotations.

Usage:
    python main.py
"""
import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from version_info import VERSION


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Beaver for YoloV8")
    app.setOrganizationName("Beaver")
    app.setApplicationVersion(VERSION)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
