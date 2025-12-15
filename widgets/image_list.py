"""
Image list widget with optimized performance for large datasets.
"""
import os
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                                QLabel, QLineEdit, QHBoxLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette


class ImageListWidget(QWidget):
    """Widget displaying a list of images (filenames only for fast loading)"""

    image_selected = Signal(int)  # Emits the index of the selected image

    def __init__(self):
        super().__init__()

        self.image_files = []
        self.images_dir = None
        self.current_index = -1

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Images")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        # Search/filter box
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search images...")
        self.filter_input.textChanged.connect(self._filter_images)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input)
        layout.addLayout(filter_layout)

        # Image count label
        self.count_label = QLabel("0 images")
        self.count_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.count_label)

        # Image list
        self.list_widget = QListWidget()
        self.list_widget.setSpacing(2)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list_widget.setStyleSheet("QListWidget::item:selected { background-color: #1e90ff; color: white; }")

        layout.addWidget(self.list_widget)

    def set_images(self, images_dir, image_files):
        """
        Set the list of images to display

        Args:
            images_dir: Directory containing the images
            image_files: List of image filenames
        """
        self.images_dir = images_dir
        self.image_files = image_files
        self.current_index = -1

        # Clear existing items
        self.list_widget.clear()
        self.filter_input.clear()

        # Update count
        self.count_label.setText(f"{len(image_files)} images")

        # Add items with just filenames (instant even with 10000+ images)
        for i, filename in enumerate(image_files):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, i)  # Store original index
            self.list_widget.addItem(item)

    def _filter_images(self, text):
        """Filter image list based on search text"""
        search_text = text.lower()

        visible_count = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                filename = item.text().lower()
                matches = search_text in filename
                item.setHidden(not matches)
                if matches:
                    visible_count += 1

        # Update count label
        if search_text:
            self.count_label.setText(f"{visible_count} of {len(self.image_files)} images")
        else:
            self.count_label.setText(f"{len(self.image_files)} images")

    def _on_item_clicked(self, item):
        """Handle item click event"""
        index = item.data(Qt.UserRole)
        self.image_selected.emit(index)

    def set_current_image(self, index):
        """
        Highlight the current image in the list

        Args:
            index: Index of the current image
        """
        self.current_index = index

        # Clear previous selection
        self.list_widget.clearSelection()

        # Find and select the item with this index
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole) == index:
                item.setSelected(True)
                self.list_widget.scrollToItem(item, QListWidget.PositionAtCenter)
                break

    def clear(self):
        """Clear the image list"""
        self.list_widget.clear()
        self.image_files = []
        self.images_dir = None
        self.current_index = -1
        self.count_label.setText("0 images")
