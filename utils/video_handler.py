"""
Video handling utilities for managing video files.
"""
import os
from typing import List
from pathlib import Path


class VideoHandler:
    """Handles file operations for videos"""

    # Supported video formats
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(self, videos_dir: str = None):
        """
        Args:
            videos_dir: Directory containing video files
        """
        self.videos_dir = videos_dir
        self.video_files: List[str] = []
        self.current_index = 0

        if videos_dir:
            self.load_video_list()

    def set_directory(self, videos_dir: str) -> None:
        """Set the videos directory"""
        self.videos_dir = videos_dir
        self.load_video_list()

    def load_video_list(self) -> None:
        """Load list of video files from the videos directory"""
        if not self.videos_dir or not os.path.exists(self.videos_dir):
            self.video_files = []
            self.current_index = 0
            return

        # Fast method: just check extension
        self.video_files = [
            file for file in sorted(os.listdir(self.videos_dir))
            if Path(file).suffix.lower() in self.VIDEO_EXTENSIONS
        ]

        self.current_index = 0 if self.video_files else 0

    def get_current_video_path(self) -> str:
        """Get the path to the current video"""
        if not self.video_files:
            return ""
        return os.path.join(self.videos_dir, self.video_files[self.current_index])

    def next_video(self) -> bool:
        """
        Move to the next video.

        Returns:
            True if successfully moved, False if at the end
        """
        if not self.video_files:
            return False

        if self.current_index < len(self.video_files) - 1:
            self.current_index += 1
            return True
        return False

    def previous_video(self) -> bool:
        """
        Move to the previous video.

        Returns:
            True if successfully moved, False if at the beginning
        """
        if not self.video_files:
            return False

        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def get_current_index(self) -> int:
        """Get the current video index (0-based)"""
        return self.current_index

    def get_total_videos(self) -> int:
        """Get the total number of videos"""
        return len(self.video_files)

    def get_current_video_name(self) -> str:
        """Get the name of the current video file"""
        if not self.video_files:
            return ""
        return self.video_files[self.current_index]

    def has_videos(self) -> bool:
        """Check if there are any videos loaded"""
        return len(self.video_files) > 0

    def get_progress_string(self) -> str:
        """Get a string representing current progress (e.g., '1/10')"""
        if not self.video_files:
            return "0/0"
        return f"{self.current_index + 1}/{len(self.video_files)}"

    def goto_video(self, index: int) -> bool:
        """
        Go to a specific video by index.

        Args:
            index: 0-based index

        Returns:
            True if successfully moved, False if index out of range
        """
        if 0 <= index < len(self.video_files):
            self.current_index = index
            return True
        return False
