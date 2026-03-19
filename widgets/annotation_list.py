"""
Widget for displaying and managing the list of annotations.
"""
from typing import List
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
                                QListWidgetItem, QPushButton, QLabel, QComboBox,
                                QGroupBox, QRadioButton, QButtonGroup)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QBrush, QPalette
from models.annotation import Annotation


class AnnotationListWidget(QWidget):
    """Widget for displaying list of annotations and class selection"""

    # Signals
    class_changed = Signal(int)  # Emitted when selected class changes
    delete_requested = Signal()  # Emitted when delete button clicked
    annotation_selected = Signal(int)  # Emitted when an annotation is clicked in the list (index)
    mode_changed = Signal(str)  # Emitted when annotation mode changes (segmentation/detection)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.class_names: List[str] = ["Class 0", "Class 1", "Class 2"]  # Default classes
        self.current_class_id: int = 0
        self.annotation_mode: str = "segmentation"

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)

        # Mode selection group
        mode_group = QGroupBox("Annotation Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_button_group = QButtonGroup(self)
        
        self.radio_segmentation = QRadioButton("Instance Segmentation")
        self.radio_segmentation.setChecked(True)
        self.radio_segmentation.toggled.connect(self._on_mode_toggled)
        self.mode_button_group.addButton(self.radio_segmentation)
        mode_layout.addWidget(self.radio_segmentation)
        
        self.radio_detection = QRadioButton("Object Detection")
        self.radio_detection.toggled.connect(self._on_mode_toggled)
        self.mode_button_group.addButton(self.radio_detection)
        mode_layout.addWidget(self.radio_detection)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Class selection group
        class_group = QGroupBox("Class Selection")
        class_layout = QVBoxLayout()

        self.class_combo = QComboBox()
        self.class_combo.addItems(self.class_names)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.class_combo.setFocusPolicy(Qt.NoFocus)
        class_layout.addWidget(QLabel("Select Class:"))
        class_layout.addWidget(self.class_combo)

        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # Annotations list group
        annotations_group = QGroupBox("Annotations")
        annotations_layout = QVBoxLayout()

        self.annotations_list = QListWidget()
        self.annotations_list.setSelectionMode(QListWidget.SingleSelection)
        self.annotations_list.itemClicked.connect(self._on_annotation_clicked)

        # Set blue highlight color for selected items using stylesheet
        self.annotations_list.setStyleSheet("""
            QListWidget::item:selected {
                background-color: rgb(0, 120, 215);
                color: white;
            }
            QListWidget::item:selected:!active {
                background-color: rgb(0, 120, 215);
                color: white;
            }
        """)

        annotations_layout.addWidget(self.annotations_list)

        # Buttons
        button_layout = QHBoxLayout()
        self.delete_button = QPushButton("Delete Selected (Delete)")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        self.delete_button.setEnabled(False)
        self.delete_button.setFocusPolicy(Qt.NoFocus)
        button_layout.addWidget(self.delete_button)

        annotations_layout.addLayout(button_layout)
        annotations_group.setLayout(annotations_layout)
        layout.addWidget(annotations_group)

        # Statistics label
        self.stats_label = QLabel("Annotations: 0")
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def set_class_names(self, class_names: List[str]):
        """Set the available class names"""
        if not class_names:
            class_names = ["Class 0"]

        self.class_names = class_names
        current_text = self.class_combo.currentText()

        self.class_combo.clear()
        self.class_combo.addItems(class_names)

        # Try to restore previous selection
        index = self.class_combo.findText(current_text)
        if index >= 0:
            self.class_combo.setCurrentIndex(index)

    def get_current_class_id(self) -> int:
        """Get the currently selected class ID"""
        return self.class_combo.currentIndex()

    def get_current_class_name(self) -> str:
        """Get the currently selected class name"""
        return self.class_combo.currentText()
        
    def get_annotation_mode(self) -> str:
        """Get the current annotation mode (segmentation or detection)"""
        return self.annotation_mode

    def set_current_class_id(self, class_id: int) -> None:
        """Set the currently selected class ID without emitting signal"""
        if 0 <= class_id < self.class_combo.count():
            # Temporarily block signals to avoid triggering class change
            self.class_combo.blockSignals(True)
            self.class_combo.setCurrentIndex(class_id)
            self.class_combo.blockSignals(False)

    def update_annotations_list(self, annotations: List[Annotation]):
        """Update the list of annotations displayed"""
        self.annotations_list.clear()

        for i, annotation in enumerate(annotations):
            class_name = annotation.class_name if annotation.class_name else f"Class {annotation.class_id}"
            item_text = f"{i + 1}. {class_name} ({len(annotation.points)} points)"

            item = QListWidgetItem(item_text)
            self.annotations_list.addItem(item)

            # Highlight selected items after adding to list
            if annotation.selected:
                self.annotations_list.setCurrentItem(item)

        # Update statistics
        self.stats_label.setText(f"Annotations: {len(annotations)}")

        # Enable/disable delete button
        self.delete_button.setEnabled(any(ann.selected for ann in annotations))

    def _on_class_changed(self, index: int):
        """Handle class selection change"""
        self.current_class_id = index
        self.class_changed.emit(index)

    def _on_mode_toggled(self):
        """Handle mode selection change"""
        if self.radio_segmentation.isChecked():
            mode = "segmentation"
        else:
            mode = "detection"
            
        if mode != self.annotation_mode:
            self.annotation_mode = mode
            self.mode_changed.emit(mode)

    def _on_delete_clicked(self):
        """Handle delete button click"""
        self.delete_requested.emit()

    def _on_annotation_clicked(self, item):
        """Handle annotation list item click"""
        index = self.annotations_list.row(item)
        self.annotation_selected.emit(index)
