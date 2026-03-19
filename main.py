"""
YOLOv8 Annotator - A GUI application for annotating images with polygon annotations.

Usage:
    python main.py
"""
import os
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow
from version_info import VERSION


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Beaver for YoloV8")
    app.setOrganizationName("Beaver")
    app.setApplicationVersion(VERSION)

    # Set application icon (window title bar and taskbar)
    icon_path = get_resource_path("beaver.jpg")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run the application event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
