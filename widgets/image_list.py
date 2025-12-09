"""
Image list widget with optimized performance for large datasets.
"""
import os
from functools import lru_cache
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                                QLabel, QLineEdit, QHBoxLayout)
from PySide6.QtCore import Qt, Signal, QSize, QThread, QObject
from PySide6.QtGui import QPixmap, QIcon, QPalette


class ThumbnailLoader(QObject):
    """Background thread worker for loading thumbnails"""
    thumbnail_ready = Signal(int, QPixmap)

    def __init__(self, image_paths, thumbnail_size):
        super().__init__()
        self.image_paths = image_paths
        self.thumbnail_size = thumbnail_size
        self.cache = {}
        self.should_stop = False

    def load_thumbnail(self, index):
        """Load thumbnail for specific index"""
        if self.should_stop:
            return

        if index in self.cache:
            self.thumbnail_ready.emit(index, self.cache[index])
            return

        if index < len(self.image_paths):
            image_path = self.image_paths[index]
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                thumbnail = pixmap.scaled(
                    self.thumbnail_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.cache[index] = thumbnail
                self.thumbnail_ready.emit(index, thumbnail)

    def clear_cache(self):
        """Clear thumbnail cache"""
        self.cache.clear()

    def stop(self):
        """Stop the loader"""
        self.should_stop = True


class ImageListWidget(QWidget):
    """Widget displaying a list of images with thumbnails"""

    image_selected = Signal(int)  # Emits the index of the selected image

    def __init__(self):
        super().__init__()

        self.image_files = []
        self.images_dir = None
        self.current_index = -1
        self.thumbnail_loader = None

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
        self.list_widget.setIconSize(QSize(80, 80))
        self.list_widget.setSpacing(2)
        self.list_widget.setWordWrap(True)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)

        # Enable smooth scrolling
        self.list_widget.verticalScrollBar().valueChanged.connect(self._on_scroll)

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

        # Stop existing thumbnail loader
        if self.thumbnail_loader:
            self.thumbnail_loader.stop()

        # Update count
        self.count_label.setText(f"{len(image_files)} images")

        # Add items without thumbnails first (fast)
        for i, filename in enumerate(image_files):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, i)  # Store original index
            self.list_widget.addItem(item)

        # Initialize thumbnail loader
        if images_dir and image_files:
            image_paths = [os.path.join(images_dir, f) for f in image_files]
            self.thumbnail_loader = ThumbnailLoader(image_paths, QSize(80, 80))
            self.thumbnail_loader.thumbnail_ready.connect(self._on_thumbnail_ready)

            # Load thumbnails for visible items
            self._load_visible_thumbnails()

    def _load_visible_thumbnails(self):
        """Load thumbnails only for currently visible items"""
        if not self.thumbnail_loader:
            return

        # Get visible range
        first_visible = self.list_widget.indexAt(self.list_widget.rect().topLeft()).row()
        last_visible = self.list_widget.indexAt(self.list_widget.rect().bottomLeft()).row()

        if first_visible < 0:
            first_visible = 0
        if last_visible < 0:
            last_visible = min(20, self.list_widget.count() - 1)  # Load first 20 initially

        # Load thumbnails for visible items + buffer
        buffer = 10
        start_idx = max(0, first_visible - buffer)
        end_idx = min(self.list_widget.count(), last_visible + buffer)

        for i in range(start_idx, end_idx):
            item = self.list_widget.item(i)
            if item:
                original_index = item.data(Qt.UserRole)
                self.thumbnail_loader.load_thumbnail(original_index)

    def _on_scroll(self):
        """Handle scroll events to load more thumbnails"""
        self._load_visible_thumbnails()

    def _on_thumbnail_ready(self, index, pixmap):
        """Handle thumbnail loaded event"""
        # Find the list item with this index
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole) == index:
                item.setIcon(QIcon(pixmap))
                break

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
        if self.thumbnail_loader:
            self.thumbnail_loader.stop()
            self.thumbnail_loader.clear_cache()

        self.list_widget.clear()
        self.image_files = []
        self.images_dir = None
        self.current_index = -1
        self.count_label.setText("0 images")
