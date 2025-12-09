"""
Main window for the YOLOv8 Annotator application.
"""
import os
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                QPushButton, QFileDialog, QStatusBar, QToolBar,
                                QMessageBox, QLabel)
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QAction, QKeySequence

from widgets.image_canvas import ImageCanvas
from widgets.annotation_list import AnnotationListWidget
from utils.file_handler import FileHandler
from utils.yolo_format import (load_annotations, save_annotations, YOLOAnnotation,
                                get_annotation_path, load_class_names, save_class_names)
from models.annotation import Annotation


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

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_shortcuts()
        self._setup_connections()

        # Install event filter to capture keyboard events globally
        self.installEventFilter(self)

        self.update_status_bar()

    def _setup_ui(self):
        """Setup the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left side: Image canvas
        self.canvas = ImageCanvas()
        self.canvas.setMinimumWidth(800)
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

        self.add_polygon_btn = QPushButton("Add Polygon")
        self.add_polygon_btn.clicked.connect(self.start_adding_polygon)
        self.add_polygon_btn.setEnabled(False)
        self.add_polygon_btn.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.add_polygon_btn)

        self.toggle_visibility_btn = QPushButton("Toggle Annotations (Space)")
        self.toggle_visibility_btn.clicked.connect(self.toggle_annotation_visibility)
        self.toggle_visibility_btn.setEnabled(False)
        self.toggle_visibility_btn.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.toggle_visibility_btn)

        self.save_button = QPushButton("Save (Ctrl+S)")
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)
        self.save_button.setFocusPolicy(Qt.NoFocus)
        action_layout.addWidget(self.save_button)

        right_layout.addLayout(action_layout)

        right_panel.setMaximumWidth(350)
        main_layout.addWidget(right_panel, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

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

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        delete_action = QAction("Delete Selected", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self.delete_selected_annotation)
        edit_menu.addAction(delete_action)

        # View menu
        view_menu = menubar.addMenu("View")

        toggle_action = QAction("Toggle Annotations", self)
        toggle_action.setShortcut(Qt.Key_Space)
        toggle_action.triggered.connect(self.toggle_annotation_visibility)
        view_menu.addAction(toggle_action)

    def _setup_toolbar(self):
        """Setup the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # Add polygon action
        add_polygon_action = QAction("Add Polygon", self)
        add_polygon_action.triggered.connect(self.start_adding_polygon)
        toolbar.addAction(add_polygon_action)

        toolbar.addSeparator()

        # Save action
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.save_annotations)
        toolbar.addAction(save_action)

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

    def select_images_folder(self):
        """Open dialog to select images folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder:
            self.file_handler.images_dir = folder
            self.images_folder_label.setText(f"Images Folder: {folder}")

            # If labels folder is also set, load the first image
            if self.file_handler.labels_dir:
                self.file_handler.set_directories(folder, self.file_handler.labels_dir)
                self.load_current_image()
                self.load_class_names_from_file()

    def select_labels_folder(self):
        """Open dialog to select labels folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Labels Folder")
        if folder:
            self.file_handler.labels_dir = folder
            self.labels_folder_label.setText(f"Labels Folder: {folder}")

            # If images folder is also set, load the first image
            if self.file_handler.images_dir:
                self.file_handler.set_directories(self.file_handler.images_dir, folder)
                self.load_current_image()
                self.load_class_names_from_file()

    def load_class_names_from_file(self):
        """Load class names from classes.txt in labels folder"""
        if not self.file_handler.labels_dir:
            return

        classes_file = os.path.join(self.file_handler.labels_dir, "classes.txt")
        if os.path.exists(classes_file):
            class_names = load_class_names(classes_file)
            if class_names:
                self.class_names = class_names
                self.annotation_widget.set_class_names(class_names)
        else:
            # Create default classes.txt
            save_class_names(classes_file, self.class_names)

    def prompt_save_changes(self):
        """Prompt user to save changes. Returns True if should continue, False if cancelled"""
        if not self.has_unsaved_changes:
            return True

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Save changes?")
        msg_box.setText("Save changes?")
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
        self.annotation_widget.update_annotations_list(self.current_annotations)
        self.status_bar.showMessage("Annotation added", 2000)

    def on_annotation_modified(self):
        """Handle annotation modified event"""
        self.has_unsaved_changes = True
        self.annotation_widget.update_annotations_list(self.current_annotations)

    def on_annotation_deleted(self, annotation):
        """Handle annotation deleted event"""
        self.has_unsaved_changes = True
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
            self.status_bar.showMessage(f"Changed annotation class to {class_name}", 2000)

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
        if event.type() == QEvent.KeyPress:
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
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key_Space:
            self.toggle_annotation_visibility()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event"""
        if not self.prompt_save_changes():
            event.ignore()
        else:
            event.accept()
