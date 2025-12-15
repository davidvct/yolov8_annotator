"""
Video list widget for browsing video files.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
                                QLabel, QLineEdit, QHBoxLayout)
from PySide6.QtCore import Qt, Signal


class VideoListWidget(QWidget):
    """Widget displaying a list of videos (filenames only for fast loading)"""

    video_selected = Signal(int)  # Emits the index of the selected video

    def __init__(self):
        super().__init__()

        self.video_files = []
        self.videos_dir = None
        self.current_index = -1

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Videos")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        # Search/filter box
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search videos...")
        self.filter_input.textChanged.connect(self._filter_videos)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input)
        layout.addLayout(filter_layout)

        # Video count label
        self.count_label = QLabel("0 videos")
        self.count_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.count_label)

        # Video list
        self.list_widget = QListWidget()
        self.list_widget.setSpacing(2)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.list_widget.setStyleSheet("QListWidget::item:selected { background-color: #1e90ff; color: white; }")

        layout.addWidget(self.list_widget)

    def set_videos(self, videos_dir, video_files):
        """
        Set the list of videos to display

        Args:
            videos_dir: Directory containing the videos
            video_files: List of video filenames
        """
        self.videos_dir = videos_dir
        self.video_files = video_files
        self.current_index = -1

        # Clear existing items
        self.list_widget.clear()
        self.filter_input.clear()

        # Update count
        self.count_label.setText(f"{len(video_files)} videos")

        # Add items with just filenames
        for i, filename in enumerate(video_files):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, i)  # Store original index
            self.list_widget.addItem(item)

    def _filter_videos(self, text):
        """Filter video list based on search text"""
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
            self.count_label.setText(f"{visible_count} of {len(self.video_files)} videos")
        else:
            self.count_label.setText(f"{len(self.video_files)} videos")

    def _on_item_clicked(self, item):
        """Handle item click event"""
        index = item.data(Qt.UserRole)
        self.video_selected.emit(index)

    def set_current_video(self, index):
        """
        Highlight the current video in the list

        Args:
            index: Index of the current video
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
        """Clear the video list"""
        self.list_widget.clear()
        self.video_files = []
        self.videos_dir = None
        self.current_index = -1
        self.count_label.setText("0 videos")
