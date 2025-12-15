"""
Main window for the YOLOv8 Annotator application.
"""
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QPushButton, QFileDialog, QStatusBar, QToolBar,
                                QMessageBox, QLabel, QTabWidget)
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QAction, QKeySequence

from widgets.image_canvas import ImageCanvas
from widgets.annotation_list import AnnotationListWidget
from widgets.image_list import ImageListWidget
from widgets.video_inference_tab import VideoInferenceTab
from utils.file_handler import FileHandler
from utils.yolo_format import (load_annotations, save_annotations, YOLOAnnotation,
                                get_annotation_path, load_class_names, save_class_names)
from utils.session_manager import SessionManager
from models.annotation import Annotation
from utils.undo_redo import UndoRedoManager


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv8 Annotator")
        self.setGeometry(100, 100, 1400, 900)

        # File handler
        self.file_handler = FileHandler()

        # Class names
        self.class_names = ["Class 0", "Class 1", "Class 2"]

        # Current annotations (in memory)
        self.current_annotations = []

        # Unsaved changes flag
        self.has_unsaved_changes = False

        # Undo/Redo manager
        self.undo_manager = UndoRedoManager(max_history=50)

        # Session tracking
        self.current_session_path = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_shortcuts()
        self._setup_connections()

        # Install event filter to capture keyboard events globally
        self.installEventFilter(self)

        self.update_status_bar()

        # Auto-load last session if available
        self._auto_load_last_session()

        # Track the last saved/loaded session state to detect changes
        self.last_saved_session_data = self._collect_session_data()

    def _setup_ui(self):
        """Setup the user interface"""
        # Create tab widget as central widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create annotation tab
        annotation_tab = self._create_annotation_tab()
        self.tab_widget.addTab(annotation_tab, "Annotation")

        # Create video inference tab
        self.video_inference_tab = VideoInferenceTab()
        self.tab_widget.addTab(self.video_inference_tab, "Video Inference")

        # Connect tab change event
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _create_annotation_tab(self):
        """Create the annotation tab content"""
        # Tab widget
        tab_widget = QWidget()
        main_layout = QHBoxLayout(tab_widget)

        # Left side: Image list
        self.image_list_widget = ImageListWidget()
        self.image_list_widget.setMaximumWidth(250)
        self.image_list_widget.setMinimumWidth(200)
        main_layout.addWidget(self.image_list_widget, stretch=0)

        # Center: Image canvas
        self.canvas = ImageCanvas()
        self.canvas.setMinimumWidth(600)
        main_layout.addWidget(self.canvas, stretch=3)

        # Right side: Annotation controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Folder selection
        folder_group = QWidget()
        folder_layout = QVBoxLayout(folder_group)

        self.images_folder_label = QLabel("Images Folder: Not selected")
        self.images_folder_label.setWordWrap(True)
        folder_layout.addWidget(self.images_folder_label)

        select_images_btn = QPushButton("Select Images Folder")
        select_images_btn.clicked.connect(self.select_images_folder)
        select_images_btn.setFocusPolicy(Qt.NoFocus)
        folder_layout.addWidget(select_images_btn)

        self.labels_folder_label = QLabel("Labels Folder: Not selected")
        self.labels_folder_label.setWordWrap(True)
        folder_layout.addWidget(self.labels_folder_label)

        select_labels_btn = QPushButton("Select Labels Folder")
        select_labels_btn.clicked.connect(self.select_labels_folder)
        select_labels_btn.setFocusPolicy(Qt.NoFocus)
        folder_layout.addWidget(select_labels_btn)

        right_layout.addWidget(folder_group)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous (←)")
        self.prev_button.clicked.connect(self.previous_image)
        self.prev_button.setEnabled(False)
        self.prev_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from capturing arrow keys

        self.next_button = QPushButton("Next (→)")
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setEnabled(False)
        self.next_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from capturing arrow keys

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        right_layout.addLayout(nav_layout)

        # Annotation list widget
        self.annotation_widget = AnnotationListWidget()
        right_layout.addWidget(self.annotation_widget)

        # Action buttons
        action_layout = QVBoxLayout()

        # Undo button
        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.undo_btn)

        # Redo button
        self.redo_btn = QPushButton("Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)
        self.redo_btn.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.redo_btn)

        self.add_polygon_btn = QPushButton("Add Polygon (hold Shift key)")
        self.add_polygon_btn.clicked.connect(self.start_adding_polygon)
        self.add_polygon_btn.setEnabled(False)
        self.add_polygon_btn.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.add_polygon_btn)

        self.toggle_visibility_btn = QPushButton("Toggle Annotations (Space)")
        self.toggle_visibility_btn.clicked.connect(self.toggle_annotation_visibility)
        self.toggle_visibility_btn.setEnabled(False)
        self.toggle_visibility_btn.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.toggle_visibility_btn)

        self.save_button = QPushButton("Save Annotation (Ctrl+S)")
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)
        self.save_button.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.save_button)

        right_layout.addLayout(action_layout)

        right_panel.setMaximumWidth(350)
        main_layout.addWidget(right_panel, stretch=1)

        return tab_widget

    def _setup_menu(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_images_action = QAction("Open Images Folder...", self)
        open_images_action.triggered.connect(self.select_images_folder)
        file_menu.addAction(open_images_action)

        open_labels_action = QAction("Open Labels Folder...", self)
        open_labels_action.triggered.connect(self.select_labels_folder)
        file_menu.addAction(open_labels_action)

        file_menu.addSeparator()

        self.save_action = QAction("Save", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.triggered.connect(self.save_annotations_if_annotation_tab)
        file_menu.addAction(self.save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Session menu
        session_menu = menubar.addMenu("Session")

        new_session_action = QAction("New Session", self)
        new_session_action.triggered.connect(self.new_session)
        session_menu.addAction(new_session_action)

        session_menu.addSeparator()

        open_session_action = QAction("Open Session...", self)
        open_session_action.setShortcut("Ctrl+O")
        open_session_action.triggered.connect(self.open_session)
        session_menu.addAction(open_session_action)

        self.save_session_action = QAction("Save Session", self)
        self.save_session_action.setShortcut("Ctrl+Shift+S")
        self.save_session_action.triggered.connect(self.save_session)
        session_menu.addAction(self.save_session_action)

        save_session_as_action = QAction("Save Session As...", self)
        save_session_as_action.triggered.connect(self.save_session_as)
        session_menu.addAction(save_session_as_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self.undo_if_annotation_tab)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self.redo_if_annotation_tab)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        delete_action = QAction("Delete Selected", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self.delete_selected_annotation_if_annotation_tab)
        edit_menu.addAction(delete_action)

        # View menu
        view_menu = menubar.addMenu("View")

        toggle_action = QAction("Toggle Annotations", self)
        toggle_action.setShortcut(Qt.Key_Space)
        toggle_action.triggered.connect(self.toggle_annotation_visibility_if_annotation_tab)
        view_menu.addAction(toggle_action)

    def _setup_toolbar(self):
        """Setup the toolbar"""
        # Toolbar removed - all annotation controls are now in the Annotation tab
        pass

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Navigation shortcuts are handled in keyPressEvent
        pass

    def _setup_connections(self):
        """Setup signal-slot connections"""
        # Canvas signals
        self.canvas.annotation_added.connect(self.on_annotation_added)
        self.canvas.annotation_modified.connect(self.on_annotation_modified)
        self.canvas.annotation_deleted.connect(self.on_annotation_deleted)
        self.canvas.annotation_selected.connect(self.on_annotation_selected)

        # Annotation widget signals
        self.annotation_widget.class_changed.connect(self.on_class_changed)
        self.annotation_widget.delete_requested.connect(self.delete_selected_annotation)
        self.annotation_widget.annotation_selected.connect(self.on_list_annotation_selected)

        # Image list widget signals
        self.image_list_widget.image_selected.connect(self.on_image_list_selected)

    def select_images_folder(self):
        """Open dialog to select images folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder:
            self.file_handler.images_dir = folder
            self.images_folder_label.setText(f"Images Folder: {folder}")

            # Load image list immediately
            if self.file_handler.labels_dir:
                # Both folders set - load everything
                self.file_handler.set_directories(folder, self.file_handler.labels_dir)
                self.update_image_list()
                self.load_current_image()
                self.load_class_names_from_file()
            else:
                # Only images folder set - just show the image list
                self.file_handler.load_image_list()
                self.update_image_list()

    def select_labels_folder(self):
        """Open dialog to select labels folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Labels Folder")
        if folder:
            self.file_handler.labels_dir = folder
            self.labels_folder_label.setText(f"Labels Folder: {folder}")

            # If images folder is also set, load the first image
            if self.file_handler.images_dir:
                self.file_handler.set_directories(self.file_handler.images_dir, folder)
                if not self.image_list_widget.image_files:
                    # Image list not populated yet - do it now
                    self.update_image_list()
                self.load_current_image()
                self.load_class_names_from_file()

    def load_class_names_from_file(self):
        """Load class names from classes.txt in dataset root folder"""
        if not self.file_handler.labels_dir:
            return

        # Place classes.txt in the parent directory of labels folder
        dataset_root = os.path.dirname(self.file_handler.labels_dir)
        classes_file = os.path.join(dataset_root, "classes.txt")
        if os.path.exists(classes_file):
            class_names = load_class_names(classes_file)
            if class_names:
                self.class_names = class_names
                self.annotation_widget.set_class_names(class_names)
        else:
            # Create default classes.txt
            save_class_names(classes_file, self.class_names)

    def update_image_list(self):
        """Update the image list widget with current images"""
        if self.file_handler.has_images():
            self.image_list_widget.set_images(
                self.file_handler.images_dir,
                self.file_handler.image_files
            )

    def on_image_list_selected(self, index):
        """Handle image selection from the image list"""
        # Check for unsaved changes before switching
        if not self.prompt_save_changes():
            return

        # Navigate to the selected image
        if self.file_handler.goto_image(index):
            self.load_current_image()
            self.update_image_list_highlight()

    def prompt_save_changes(self):
        """Prompt user to save changes. Returns True if should continue, False if cancelled"""
        if not self.has_unsaved_changes:
            return True

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Save Annotation Changes?")
        msg_box.setText("Save Annotation Changes?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Yes)

        reply = msg_box.exec()

        if reply == QMessageBox.Yes:
            self.save_annotations()
            return True
        elif reply == QMessageBox.No:
            return True
        else:  # Cancel
            return False

    def load_current_image(self, check_unsaved=False):
        """Load the current image and its annotations"""
        if not self.file_handler.has_images():
            QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
            return

        # Prompt to save current annotations before loading new image
        if check_unsaved and not self.prompt_save_changes():
            return

        # Load image
        image_path = self.file_handler.get_current_image_path()
        success = self.canvas.load_image(image_path)

        if not success:
            QMessageBox.critical(self, "Error", f"Failed to load image: {image_path}")
            return

        # Load annotations
        label_path = self.file_handler.get_current_label_path()
        yolo_annotations = load_annotations(label_path)

        # Convert to Annotation objects
        self.current_annotations = []
        for yolo_ann in yolo_annotations:
            class_name = self.class_names[yolo_ann.class_id] if yolo_ann.class_id < len(self.class_names) else f"Class {yolo_ann.class_id}"
            annotation = Annotation(yolo_ann.class_id, yolo_ann.points, class_name)
            self.current_annotations.append(annotation)

        # Update canvas
        self.canvas.set_annotations(self.current_annotations)

        # Update annotation list widget
        self.annotation_widget.update_annotations_list(self.current_annotations)

        # Enable buttons
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        self.add_polygon_btn.setEnabled(True)
        self.toggle_visibility_btn.setEnabled(True)
        self.save_button.setEnabled(True)

        # Reset unsaved changes flag
        self.has_unsaved_changes = False

        # Clear undo/redo history and push initial state
        self.undo_manager.clear()
        self.undo_manager.push_state(self.current_annotations)
        self.undo_manager.mark_saved()  # Mark initial loaded state as saved
        self.update_undo_redo_actions()

        # Update image list highlighting
        self.update_image_list_highlight()

        self.update_status_bar()

    def save_annotations(self):
        """Save current annotations to file"""
        if not self.file_handler.has_images():
            return

        label_path = self.file_handler.get_current_label_path()

        # Convert Annotation objects to YOLOAnnotation
        yolo_annotations = []
        for annotation in self.current_annotations:
            yolo_ann = YOLOAnnotation(annotation.class_id, annotation.points)
            yolo_annotations.append(yolo_ann)

        # Save to file (will delete file if no annotations)
        save_annotations(label_path, yolo_annotations)

        self.has_unsaved_changes = False
        self.undo_manager.mark_saved()
        self.update_status_bar()
        self.status_bar.showMessage("Annotations saved", 2000)

    def next_image(self):
        """Navigate to the next image"""
        if not self.prompt_save_changes():
            return

        if self.file_handler.next_image():
            self.load_current_image()
        else:
            self.status_bar.showMessage("Already at the last image", 2000)

    def previous_image(self):
        """Navigate to the previous image"""
        if not self.prompt_save_changes():
            return

        if self.file_handler.previous_image():
            self.load_current_image()
        else:
            self.status_bar.showMessage("Already at the first image", 2000)

    def start_adding_polygon(self):
        """Start adding a new polygon annotation"""
        class_id = self.annotation_widget.get_current_class_id()
        class_name = self.annotation_widget.get_current_class_name()
        self.canvas.start_drawing(class_id, class_name)
        self.status_bar.showMessage("Hold Shift + Click to add points. Release Shift to finish. Press Escape to cancel.")

    def toggle_annotation_visibility(self):
        """Toggle visibility of annotations"""
        self.canvas.toggle_annotation_visibility()

    def delete_selected_annotation(self):
        """Delete the selected annotation"""
        self.canvas.delete_selected_annotation()
        self.annotation_widget.update_annotations_list(self.current_annotations)

    def on_annotation_added(self, annotation):
        """Handle annotation added event"""
        self.has_unsaved_changes = True
        self.push_undo_state()
        self.annotation_widget.update_annotations_list(self.current_annotations)
        self.status_bar.showMessage("Annotation added", 2000)

    def on_annotation_modified(self):
        """Handle annotation modified event"""
        self.has_unsaved_changes = True
        self.push_undo_state()
        self.annotation_widget.update_annotations_list(self.current_annotations)

    def on_annotation_deleted(self, annotation):
        """Handle annotation deleted event"""
        self.has_unsaved_changes = True
        self.push_undo_state()
        self.annotation_widget.update_annotations_list(self.current_annotations)
        self.status_bar.showMessage("Annotation deleted", 2000)

    def on_annotation_selected(self, annotation):
        """Handle annotation selection event from canvas"""
        # Update the class combo box to show the selected annotation's class
        self.annotation_widget.set_current_class_id(annotation.class_id)
        self.annotation_widget.update_annotations_list(self.current_annotations)

    def on_list_annotation_selected(self, index):
        """Handle annotation selection event from list widget"""
        self.canvas.select_annotation_by_index(index)

    def on_class_changed(self, class_id):
        """Handle class selection change"""
        # If there's a selected annotation, update its class
        if self.canvas.selected_annotation:
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            self.canvas.selected_annotation.update_class(class_id, class_name)
            self.canvas.redraw_annotations()
            self.annotation_widget.update_annotations_list(self.current_annotations)
            self.has_unsaved_changes = True
            self.push_undo_state()
            self.status_bar.showMessage(f"Changed annotation class to {class_name}", 2000)

    def push_undo_state(self):
        """Push the current annotation state to the undo manager"""
        self.undo_manager.push_state(self.current_annotations)
        self.update_undo_redo_actions()

    def is_annotation_tab_active(self):
        """Check if the annotation tab is currently active"""
        return self.tab_widget.currentIndex() == 0

    def undo_if_annotation_tab(self):
        """Undo only if annotation tab is active"""
        if self.is_annotation_tab_active():
            self.undo()

    def redo_if_annotation_tab(self):
        """Redo only if annotation tab is active"""
        if self.is_annotation_tab_active():
            self.redo()

    def save_annotations_if_annotation_tab(self):
        """Save annotations only if annotation tab is active"""
        if self.is_annotation_tab_active():
            self.save_annotations()

    def delete_selected_annotation_if_annotation_tab(self):
        """Delete selected annotation only if annotation tab is active"""
        if self.is_annotation_tab_active():
            self.delete_selected_annotation()

    def toggle_annotation_visibility_if_annotation_tab(self):
        """Toggle annotation visibility only if annotation tab is active"""
        if self.is_annotation_tab_active():
            self.toggle_annotation_visibility()

    def undo(self):
        """Undo the last annotation change"""
        if self.undo_manager.can_undo():
            self.current_annotations = self.undo_manager.undo()
            self.canvas.set_annotations(self.current_annotations)
            self.annotation_widget.update_annotations_list(self.current_annotations)
            self.has_unsaved_changes = not self.undo_manager.is_saved()
            self.update_undo_redo_actions()
            self.update_status_bar()
            self.status_bar.showMessage("Undo", 2000)

    def redo(self):
        """Redo the last undone annotation change"""
        if self.undo_manager.can_redo():
            self.current_annotations = self.undo_manager.redo()
            self.canvas.set_annotations(self.current_annotations)
            self.annotation_widget.update_annotations_list(self.current_annotations)
            self.has_unsaved_changes = not self.undo_manager.is_saved()
            self.update_undo_redo_actions()
            self.update_status_bar()
            self.status_bar.showMessage("Redo", 2000)

    def update_undo_redo_actions(self):
        """Update the enabled state of undo/redo actions"""
        self.undo_action.setEnabled(self.undo_manager.can_undo())
        self.redo_action.setEnabled(self.undo_manager.can_redo())
        self.undo_btn.setEnabled(self.undo_manager.can_undo())
        self.redo_btn.setEnabled(self.undo_manager.can_redo())

    def update_image_list_highlight(self):
        """Update the highlighted image in the image list"""
        if self.file_handler.has_images():
            current_index = self.file_handler.get_current_index()
            self.image_list_widget.set_current_image(current_index)

    def update_status_bar(self):
        """Update the status bar with current information"""
        if self.file_handler.has_images():
            progress = self.file_handler.get_progress_string()
            filename = self.file_handler.get_current_image_name()
            status_text = f"Image: {filename} | {progress}"
            if self.has_unsaved_changes:
                status_text += " | Unsaved changes"
            self.status_bar.showMessage(status_text)
        else:
            self.status_bar.showMessage("No images loaded")

    def eventFilter(self, obj, event):
        """Filter events to capture keyboard shortcuts globally"""
        if event.type() == QEvent.KeyPress and self.is_annotation_tab_active():
            if event.key() == Qt.Key_Right:
                self.next_image()
                return True
            elif event.key() == Qt.Key_Left:
                self.previous_image()
                return True
            elif event.key() == Qt.Key_Space:
                self.toggle_annotation_visibility()
                return True

        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Handle key press events"""
        if self.is_annotation_tab_active():
            if event.key() == Qt.Key_Right:
                self.next_image()
            elif event.key() == Qt.Key_Left:
                self.previous_image()
            elif event.key() == Qt.Key_Space:
                self.toggle_annotation_visibility()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def on_tab_changed(self, index):
        """Handle tab change events"""
        # If switching to Annotation tab (index 0)
        if index == 0:
            self.refresh_annotation_resources()

    def refresh_annotation_resources(self):
        """Refresh resources in the annotation tab (check for new files)"""
        if self.file_handler.images_dir:
            # Capture current image name before reloading
            current_image_name = self.file_handler.get_current_image_name()
            
            # Reload image list from disk
            self.file_handler.load_image_list()
            self.update_image_list()
            
            # Try to restore the previous image selection
            image_found = False
            if current_image_name and self.file_handler.image_files:
                try:
                    new_index = self.file_handler.image_files.index(current_image_name)
                    self.file_handler.goto_image(new_index)
                    image_found = True
                except ValueError:
                    image_found = False
            
            # If image not found or no previous image, reset to 0 (default behavior of load_image_list)
            # But we need to ensure the UI reflects this change if we were previously looking at a valid image
            if not image_found:
                 self.file_handler.current_index = 0
                 if self.file_handler.has_images():
                     self.load_current_image()
            
            # Always update highlight
            self.update_image_list_highlight()

            # Refresh classes if labels dir is set
            if self.file_handler.labels_dir:
                self.load_class_names_from_file()

    # Session management methods

    def _collect_session_data(self) -> dict:
        """Collect current application state for session saving."""
        return {
            "version": "1.0",
            "annotation_tab": {
                "images_folder": self.file_handler.images_dir,
                "labels_folder": self.file_handler.labels_dir,
                "current_image_index": self.file_handler.get_current_index()
            },
            "video_tab": self.video_inference_tab.get_session_state()
        }

    def _restore_session_data(self, session_data: dict) -> list:
        """
        Restore application state from session data.

        Returns:
            List of warning messages for paths that don't exist
        """
        warnings = []

        # Restore annotation tab state
        annotation_data = session_data.get("annotation_tab", {})

        images_folder = annotation_data.get("images_folder")
        labels_folder = annotation_data.get("labels_folder")

        # Restore images folder
        if images_folder:
            if os.path.exists(images_folder):
                self.file_handler.images_dir = images_folder
                self.images_folder_label.setText(f"Images Folder: {images_folder}")
            else:
                warnings.append(f"Images folder not found: {images_folder}")
                images_folder = None

        # Restore labels folder
        if labels_folder:
            if os.path.exists(labels_folder):
                self.file_handler.labels_dir = labels_folder
                self.labels_folder_label.setText(f"Labels Folder: {labels_folder}")
            else:
                warnings.append(f"Labels folder not found: {labels_folder}")
                labels_folder = None

        # If both folders exist, load the image list and navigate to saved index
        if images_folder and labels_folder:
            self.file_handler.set_directories(images_folder, labels_folder)
            self.update_image_list()
            self.load_class_names_from_file()

            # Navigate to saved image index
            image_index = annotation_data.get("current_image_index", 0)
            if self.file_handler.has_images():
                if image_index < len(self.file_handler.image_files):
                    self.file_handler.goto_image(image_index)
                self.load_current_image()

        # Restore video tab state
        video_data = session_data.get("video_tab", {})
        video_warnings = self.video_inference_tab.restore_session_state(video_data)
        warnings.extend(video_warnings)

        return warnings

    def _update_window_title(self):
        """Update window title to show session filename."""
        if self.current_session_path:
            session_name = os.path.basename(self.current_session_path)
            self.setWindowTitle(f"YOLOv8 Annotator - {session_name}")
        else:
            self.setWindowTitle("YOLOv8 Annotator")

    def _reset_to_defaults(self):
        """Reset application to default state."""
        # Reset annotation tab
        self.file_handler.images_dir = None
        self.file_handler.labels_dir = None
        self.file_handler.image_files = []
        self.file_handler.current_index = 0

        self.images_folder_label.setText("Images Folder: Not selected")
        self.labels_folder_label.setText("Labels Folder: Not selected")
        self.image_list_widget.clear()
        self.canvas.scene.clear()
        self.canvas.pixmap_item = None
        self.canvas.annotations = []
        self.canvas.polygon_items.clear()
        self.canvas.vertex_items.clear()
        self.current_annotations = []
        self.annotation_widget.update_annotations_list([])
        self.has_unsaved_changes = False
        self.undo_manager.clear()
        
        # Reset last saved session data to defaults
        self.last_saved_session_data = self._collect_session_data()

        # Disable buttons
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.add_polygon_btn.setEnabled(False)
        self.toggle_visibility_btn.setEnabled(False)
        self.save_button.setEnabled(False)
        self.update_undo_redo_actions()

        # Reset video tab - restore default session values
        default_video_state = SessionManager.get_default_session()["video_tab"]
        self.video_inference_tab.restore_session_state(default_video_state)

        # Clear session path
        self.current_session_path = None
        self._update_window_title()

        self.update_status_bar()

    def _auto_load_last_session(self):
        """Auto-load the last opened session on startup."""
        last_session_path = SessionManager.get_last_session_path()
        if last_session_path:
            session_data = SessionManager.load_session(last_session_path)
            if session_data is not None:
                # Restore session without resetting first (already at defaults)
                warnings = self._restore_session_data(session_data)
                self.current_session_path = last_session_path
                self._update_window_title()

                # Show warnings if any paths were missing
                if warnings:
                    warning_text = "Last session loaded with warnings:\n\n" + "\n".join(f"- {w}" for w in warnings)
                    QMessageBox.warning(self, "Session Loaded with Warnings", warning_text)
                else:
                    self.status_bar.showMessage(f"Loaded last session: {os.path.basename(last_session_path)}", 3000)

    def new_session(self):
        """Create a new session (reset to defaults)."""
        # Prompt to save current annotation changes
        if not self.prompt_save_changes():
            return

        self._reset_to_defaults()
        # Clear the last session path since this is a new session
        SessionManager.clear_last_session_path()
        self.status_bar.showMessage("New session created", 2000)

    def open_session(self):
        """Open a session file."""
        # Prompt to save current annotation changes
        if not self.prompt_save_changes():
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Session",
            "",
            "Session Files (*.json);;All Files (*)"
        )

        if not filepath:
            return

        session_data = SessionManager.load_session(filepath)
        if session_data is None:
            QMessageBox.critical(self, "Error", "Failed to load session file. The file may be corrupted or in an invalid format.")
            return

        # Reset before restoring
        self._reset_to_defaults()

        # Restore session
        warnings = self._restore_session_data(session_data)

        # Update session path and title
        self.current_session_path = filepath
        self._update_window_title()

        # Save as last session
        SessionManager.save_last_session_path(filepath)

        # Show warnings if any paths were missing
        if warnings:
            warning_text = "Some paths could not be restored:\n\n" + "\n".join(f"- {w}" for w in warnings)
            QMessageBox.warning(self, "Session Loaded with Warnings", warning_text)
        else:
            self.status_bar.showMessage(f"Session loaded: {os.path.basename(filepath)}", 2000)
            
        # Update last saved state
        self.last_saved_session_data = self._collect_session_data()

    def save_session(self):
        """Save current session to file."""
        if self.current_session_path:
            self._save_session_to_file(self.current_session_path)
        else:
            self.save_session_as()

    def save_session_as(self):
        """Save session to a new file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session As",
            "",
            "Session Files (*.json);;All Files (*)"
        )

        if not filepath:
            return

        # Ensure .json extension
        if not filepath.lower().endswith('.json'):
            filepath += '.json'

        self._save_session_to_file(filepath)

    def _save_session_to_file(self, filepath: str):
        """Save session data to the specified file."""
        session_data = self._collect_session_data()
        success = SessionManager.save_session(filepath, session_data)

        if success:
            self.current_session_path = filepath
            self._update_window_title()
            # Save as last session
            SessionManager.save_last_session_path(filepath)
            self.status_bar.showMessage(f"Session saved: {os.path.basename(filepath)}", 2000)
            
            # Update last saved state
            self.last_saved_session_data = self._collect_session_data()
        else:
            QMessageBox.critical(self, "Error", "Failed to save session file.")

    def closeEvent(self, event):
        """Handle window close event"""
        # 1. Check for unsaved ANNOTATION changes
        if not self.prompt_save_changes():
            event.ignore()
            return

        # 2. Check for unsaved SESSION changes
        current_session_data = self._collect_session_data()
        
        # Compare just the important parts, or the whole dict.
        # Since _collect_session_data returns dictionaries, simple equality check should work 
        # as long as the order of items is consistent (standard dicts in recent Python are).
        # To be safe, we can rely on standard dict equality which checks contents.
        
        if current_session_data != self.last_saved_session_data:
            # Check if session is "empty" (just defaults) and we never saved it.
            # If so, maybe we don't need to prompt? 
            # But the user might have set up folders and wants to save.
            # So always prompt if different from last saved/loaded state.
            
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Save Session Changes?")
            msg_box.setText("You have unsaved session changes (folders, settings, etc.). Do you want to save the session?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            msg_box.setDefaultButton(QMessageBox.Yes)
            
            reply = msg_box.exec()
            
            if reply == QMessageBox.Yes:
                self.save_session()
                # If save was cancelled inside save_session (e.g. file dialog cancel),
                # we should probably verify if it was saved.
                # However, save_session logic above doesn't return boolean clearly to here easily 
                # without refactoring save_session.
                # Let's check if the state matches now.
                if self._collect_session_data() == self.last_saved_session_data:
                    event.accept()
                else:
                    # User cancelled save or save failed
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
            # If No, just continue to close
            
        event.accept()
