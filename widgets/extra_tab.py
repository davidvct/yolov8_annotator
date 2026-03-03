import os
import shutil
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QLineEdit, QPushButton, QFileDialog,
                               QRadioButton, QButtonGroup, QTextBrowser, 
                               QGroupBox, QFormLayout, QCheckBox, QMessageBox,
                               QComboBox)
from PySide6.QtCore import Qt, QThread, Signal
from PIL import Image

class FilterWorker(QThread):
    """Worker thread for filtering and copying/moving files to avoid UI freezing."""
    log_msg = Signal(str)
    progress_update = Signal(int, int) # current, total
    finished = Signal(bool, str) # success, message

    def __init__(self, img_dir, mask_dir, target_dir, filter_word, is_move, case_sensitive, match_exact, target_ext, target_type, class_remapping, replace_word_enabled=False, replace_word=""):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_dir = target_dir
        self.filter_word = filter_word
        self.is_move = is_move
        self.case_sensitive = case_sensitive
        self.match_exact = match_exact
        self.target_ext = target_ext
        self.target_type = target_type
        self.class_remapping = class_remapping
        self.replace_word_enabled = replace_word_enabled
        self.replace_word = replace_word

    def run(self):
        try:
            # 1. Validate directories
            if not self.img_dir and not self.mask_dir:
                self.finished.emit(False, "At least one of Image or Mask directory must be specified.")
                return
                
            if self.img_dir and not os.path.isdir(self.img_dir):
                self.finished.emit(False, f"Image directory not found: {self.img_dir}")
                return
                
            if self.mask_dir and not os.path.isdir(self.mask_dir):
                self.finished.emit(False, f"Mask directory not found: {self.mask_dir}")
                return
            
            # Create target dir if it doesn't exist
            if not os.path.exists(self.target_dir):
                os.makedirs(self.target_dir)
                self.log_msg.emit(f"Created target directory: {self.target_dir}")
            elif not os.path.isdir(self.target_dir):
                 self.finished.emit(False, f"Target path is not a directory: {self.target_dir}")
                 return

            # 2. Find matching files based on what directories are provided
            matches = []  # List of tuples: (base_name, img_filename/None, mask_filename/None)
            filter_lower = self.filter_word.lower() if not self.case_sensitive else self.filter_word

            def is_match(filename):
                base_name = os.path.splitext(filename)[0]
                # If exact, it should match the filename without extension exactly
                target = base_name if self.case_sensitive else base_name.lower()
                
                if self.match_exact:
                    # For exact match with empty filter_word, process all files
                    if not filter_lower:
                        return True
                    return filter_lower == target
                else:
                    if not filter_lower:
                        return True
                    return filter_lower in target

            # If image directory is provided, scan it
            if self.img_dir:
                for f in os.listdir(self.img_dir):
                    if self.target_type == "images":
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.tif')):
                            if is_match(f):
                                base_name = os.path.splitext(f)[0]
                                # Assume mask might exist if mask dir is provided
                                mask_filename = base_name + '.png' if self.mask_dir else None
                                matches.append((base_name, f, mask_filename))
                    elif self.target_type == "text":
                        if f.lower().endswith('.txt'):
                            if is_match(f):
                                base_name = os.path.splitext(f)[0]
                                matches.append((base_name, f, None))
                            
            # If ONLY mask directory is provided, scan it directly
            elif self.mask_dir:
                 for f in os.listdir(self.mask_dir):
                    if f.lower().endswith('.png'):
                        if is_match(f):
                            base_name = os.path.splitext(f)[0]
                            matches.append((base_name, "", f))

            if not matches:
                self.finished.emit(True, f"No matching files found for '{self.filter_word}'.")
                return

            self.log_msg.emit(f"Found {len(matches)} matching base filenames. Checking corresponding files...")
            
            # 3. Process each match
            success_count = 0
            missing_files = 0
            total_items = len(matches)
            
            for i, (base_name, img_filename, mask_filename) in enumerate(matches):
                # Resolve paths
                img_path = os.path.join(self.img_dir, img_filename) if self.img_dir and img_filename else None
                mask_path = os.path.join(self.mask_dir, mask_filename) if self.mask_dir and mask_filename else None
                
                new_base_name = base_name
                if self.replace_word_enabled and self.filter_word:
                    if self.case_sensitive:
                        new_base_name = base_name.replace(self.filter_word, self.replace_word)
                    else:
                        import re
                        new_base_name = re.compile(re.escape(self.filter_word), re.IGNORECASE).sub(self.replace_word, base_name)

                if mask_filename:
                    _, mask_ext = os.path.splitext(mask_filename)
                    target_mask_path = os.path.join(self.target_dir, f"{new_base_name}{mask_ext}")
                else:
                    target_mask_path = None

                # Track if at least one file for the base_name was successful
                item_success = False

                # Process Image or Text
                if img_path and img_filename:
                    if self.target_type == "text":
                        _, current_ext_with_dot = os.path.splitext(img_filename)
                        target_txt_path = os.path.join(self.target_dir, f"{new_base_name}{current_ext_with_dot}")
                        if os.path.exists(img_path):
                            try:
                                if not self.class_remapping:
                                    # Just copy/move if no remapping provided
                                    if self.is_move:
                                        shutil.move(img_path, target_txt_path)
                                    else:
                                        shutil.copy2(img_path, target_txt_path)
                                else:
                                    # Apply remapping logic
                                    try:
                                        import ast
                                        remap_list = ast.literal_eval(self.class_remapping)
                                        remap_map = dict(remap_list) # {src: dst}
                                    except Exception as e:
                                        self.log_msg.emit(f"Invalid class remapping format: {self.class_remapping}. Error: {e}")
                                        remap_map = {}

                                    if remap_map:
                                        keep_set = set(remap_map.keys())
                                        kept_lines = []
                                        skipped = 0
                                        with open(img_path, "r") as f:
                                            for line in f:
                                                stripped = line.strip()
                                                if not stripped:
                                                    continue
                                                tokens = [t.strip() for t in stripped.replace(",", " ").split()]
                                                if not tokens:
                                                    continue
                                                try:
                                                    class_idx = int(tokens[0])
                                                except ValueError:
                                                    continue
                                                
                                                if class_idx not in keep_set:
                                                    skipped += 1
                                                    continue
                                                
                                                out_idx = remap_map.get(class_idx, class_idx)
                                                out_tokens = [str(out_idx)] + tokens[1:]
                                                kept_lines.append(" ".join(out_tokens))
                                        
                                        with open(target_txt_path, "w") as f:
                                            f.write("\n".join(kept_lines))
                                            if kept_lines:
                                                f.write("\n")
                                        
                                        if self.is_move:
                                            os.remove(img_path)
                                        self.log_msg.emit(f"Processed {img_filename}: kept {len(kept_lines)}, skipped {skipped}")
                                    else:
                                        # Fallback if empty remap_map
                                        if self.is_move:
                                            shutil.move(img_path, target_txt_path)
                                        else:
                                            shutil.copy2(img_path, target_txt_path)

                                item_success = True
                            except Exception as e:
                                self.log_msg.emit(f"Error processing text file {img_filename}: {str(e)}")
                        else:
                            self.log_msg.emit(f"Warning: Expected text file not found: {img_path}")
                            missing_files += 1

                    else:
                        # Process Image
                        # Determine source and target extensions for the image
                        _, current_ext_with_dot = os.path.splitext(img_filename)
                        current_ext = current_ext_with_dot.lower().strip('.')
                        
                        if self.target_ext == "same as source":
                            actual_target_ext = current_ext
                            target_img_filename = f"{new_base_name}{current_ext_with_dot}"
                        else:
                            actual_target_ext = self.target_ext
                            target_img_filename = f"{new_base_name}.{actual_target_ext}"

                        target_img_path = os.path.join(self.target_dir, target_img_filename)

                        if os.path.exists(img_path):
                            try:
                                if actual_target_ext == current_ext:
                                    # No conversion needed
                                    if self.is_move:
                                        shutil.move(img_path, target_img_path)
                                    else:
                                        shutil.copy2(img_path, target_img_path)
                                else:
                                    # Conversion needed
                                    with Image.open(img_path) as img:
                                        if actual_target_ext == 'jpg':
                                            if img.mode != 'RGB':
                                                img = img.convert('RGB')
                                            img.save(target_img_path, "JPEG", quality=90)
                                        elif actual_target_ext == 'png':
                                            img.save(target_img_path, "PNG")
                                        else:
                                            # Fallback
                                            img.save(target_img_path)
                                    
                                    if self.is_move:
                                        os.remove(img_path)

                                item_success = True
                            except Exception as e:
                                self.log_msg.emit(f"Error processing image {img_filename}: {str(e)}")
                        else:
                            self.log_msg.emit(f"Warning: Expected image not found: {img_path}")
                            missing_files += 1

                # Process Mask
                if mask_path and target_mask_path and mask_filename:
                    if os.path.exists(mask_path):
                        try:
                            if self.is_move:
                                shutil.move(mask_path, target_mask_path)
                            else:
                                shutil.copy2(mask_path, target_mask_path)
                            item_success = True
                        except Exception as e:
                            self.log_msg.emit(f"Error processing mask {mask_filename}: {str(e)}")
                    else:
                        if self.img_dir: # Only warn about missing masks if we found an image pair
                            self.log_msg.emit(f"Warning: Expected mask not found: {mask_path}")
                            missing_files += 1
                        
                if item_success:
                    success_count += 1
                
                self.progress_update.emit(i + 1, total_items)

            op_name = "moved" if self.is_move else "copied"
            msg = f"Operation complete. Successfully {op_name} {success_count} item groups."
            if missing_files > 0:
                msg += f" ({missing_files} expected files were missing)"
                
            self.finished.emit(True, msg)

        except Exception as e:
             self.finished.emit(False, f"An unexpected error occurred: {str(e)}")


class ExtraTab(QWidget):
    """Extra tab for filtering and moving/copying image and mask pairs."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.worker = None

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Directories Selection Group
        dir_group = QGroupBox("Directories")
        dir_layout = QFormLayout(dir_group)

        # Image Directory
        img_container = QVBoxLayout()
        img_row = QHBoxLayout()
        self.img_dir_input = QLineEdit()
        self.img_dir_input.setPlaceholderText("Select folder containing images (optional)...")
        img_browse_btn = QPushButton("Browse...")
        img_browse_btn.clicked.connect(lambda: self._browse_folder(self.img_dir_input, "Select Image Directory"))
        img_row.addWidget(self.img_dir_input)
        img_row.addWidget(img_browse_btn)
        img_container.addLayout(img_row)
        
        img_lbl = QLabel("Supported extensions: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp, .txt")
        img_lbl.setStyleSheet("color: gray; font-size: 11px;")
        img_container.addWidget(img_lbl)
        
        dir_layout.addRow("Source Dir:", img_container)

        # Mask Directory
        mask_container = QVBoxLayout()
        mask_row = QHBoxLayout()
        self.mask_dir_input = QLineEdit()
        self.mask_dir_input.setPlaceholderText("Select folder containing masks (optional)...")
        self.mask_dir_input.setEnabled(False)
        self.mask_browse_btn = QPushButton("Browse...")
        self.mask_browse_btn.setEnabled(False)
        self.mask_browse_btn.clicked.connect(lambda: self._browse_folder(self.mask_dir_input, "Select Mask Directory"))
        mask_row.addWidget(self.mask_dir_input)
        mask_row.addWidget(self.mask_browse_btn)
        mask_container.addLayout(mask_row)
        
        mask_lbl = QLabel("Supported extensions: .png")
        mask_lbl.setStyleSheet("color: gray; font-size: 11px;")
        mask_container.addWidget(mask_lbl)
        
        self.mask_enable_cb = QCheckBox("Mask Path:")
        self.mask_enable_cb.setChecked(False)
        self.mask_enable_cb.toggled.connect(self._toggle_mask_input)
        
        dir_layout.addRow(self.mask_enable_cb, mask_container)

        # Target Directory
        target_row = QHBoxLayout()
        self.target_dir_input = QLineEdit()
        self.target_dir_input.setPlaceholderText("Select destination folder...")
        target_browse_btn = QPushButton("Browse...")
        target_browse_btn.clicked.connect(lambda: self._browse_folder(self.target_dir_input, "Select Target Destination"))
        target_row.addWidget(self.target_dir_input)
        target_row.addWidget(target_browse_btn)
        dir_layout.addRow("Target Destination:", target_row)

        main_layout.addWidget(dir_group)

        # 2. Filter Options Group
        filter_group = QGroupBox("Filter Options")
        filter_layout = QVBoxLayout(filter_group)

        # Target Type Selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Target Type:"))
        self.target_type_combo = QComboBox()
        self.target_type_combo.addItems(["images", "text"])
        self.target_type_combo.currentTextChanged.connect(self._on_target_type_changed)
        type_layout.addWidget(self.target_type_combo)
        type_layout.addStretch()
        filter_layout.addLayout(type_layout)

        # Filter Word Input
        word_layout = QHBoxLayout()
        word_layout.addWidget(QLabel("Find filename with word:"))
        self.filter_word_input = QLineEdit()
        self.filter_word_input.setPlaceholderText("Enter substring or exact match. Leave empty for all.")
        word_layout.addWidget(self.filter_word_input)
        
        self.replace_cb = QCheckBox("Replace word with:")
        self.replace_cb.toggled.connect(self._toggle_replace_input)
        self.replace_input = QLineEdit()
        self.replace_input.setEnabled(False)
        self.replace_input.setPlaceholderText("Replace matched word with...")
        word_layout.addWidget(self.replace_cb)
        word_layout.addWidget(self.replace_input)
        
        filter_layout.addLayout(word_layout)

        # Class Remapping (for text type)
        self.class_remap_layout = QHBoxLayout()
        self.class_remap_layout.addWidget(QLabel("Class Remapping:"))
        self.class_remap_input = QLineEdit()
        self.class_remap_input.setPlaceholderText("e.g., [(1,3), (2,2)] to keep class 1 as 3, class 2 as 2")
        self.class_remap_input.setEnabled(False) # Default disabled until text is selected
        self.class_remap_layout.addWidget(self.class_remap_input)
        filter_layout.addLayout(self.class_remap_layout)

        # Operation Selection
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Operation:"))
        self.copy_radio = QRadioButton("Copy")
        self.copy_radio.setChecked(True)
        self.move_radio = QRadioButton("Move")
        
        self.op_group = QButtonGroup(self)
        self.op_group.addButton(self.copy_radio)
        self.op_group.addButton(self.move_radio)
        
        op_layout.addWidget(self.copy_radio)
        op_layout.addWidget(self.move_radio)
        
        op_layout.addSpacing(20)
        
        self.ext_label = QLabel("Convert Ext to:")
        op_layout.addWidget(self.ext_label)
        self.ext_combo = QComboBox()
        self.ext_combo.addItems(["same as source", "jpg", "png"])
        op_layout.addWidget(self.ext_combo)
        
        op_layout.addSpacing(20)
        self.case_sensitive_cb = QCheckBox("Case Sensitive")
        self.match_exact_cb = QCheckBox("Match Exact")
        op_layout.addWidget(self.case_sensitive_cb)
        op_layout.addWidget(self.match_exact_cb)
        
        op_layout.addStretch()
        filter_layout.addLayout(op_layout)

        # Action Button
        action_layout = QHBoxLayout()
        self.filter_btn = QPushButton("Filter and Execute")
        self.filter_btn.setMinimumHeight(40)
        self.filter_btn.clicked.connect(self._run_filter)
        action_layout.addStretch()
        action_layout.addWidget(self.filter_btn)
        action_layout.addStretch()
        filter_layout.addLayout(action_layout)

        main_layout.addWidget(filter_group)

        # 3. Log Output
        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout(log_group)
        self.log_browser = QTextBrowser()
        log_layout.addWidget(self.log_browser)
        main_layout.addWidget(log_group)

    def _toggle_mask_input(self, checked):
        """Enable or disable the mask inputs."""
        self.mask_dir_input.setEnabled(checked)
        self.mask_browse_btn.setEnabled(checked)

    def _on_target_type_changed(self, target_type):
        """Update UI based on target type selection (images vs text)."""
        if target_type == "text":
            self.ext_combo.setEnabled(False)
            self.ext_label.setStyleSheet("color: gray;")
            self.class_remap_input.setEnabled(True)
            self.mask_enable_cb.setChecked(False)
            self.mask_enable_cb.setEnabled(False)
            self.log_browser.append("Info: Text type selected. Ext conversion disabled.")
        else:
            self.ext_combo.setEnabled(True)
            self.ext_label.setStyleSheet("")
            self.class_remap_input.setEnabled(False)
            self.mask_enable_cb.setEnabled(True)

    def _toggle_replace_input(self, checked):
        """Enable or disable the replace word input."""
        self.replace_input.setEnabled(checked)

    def _browse_folder(self, line_edit, title):
        """Open a directory selection dialog and update the line edit."""
        folder = QFileDialog.getExistingDirectory(self, title)
        if folder:
            line_edit.setText(folder)

    def _log(self, text):
        """Append text to the log browser."""
        self.log_browser.append(text)

    def _run_filter(self):
        """Validate inputs and start the worker thread."""
        img_dir = self.img_dir_input.text().strip()
        mask_dir = self.mask_dir_input.text().strip() if self.mask_enable_cb.isChecked() else ""
        target_dir = self.target_dir_input.text().strip()
        filter_word = self.filter_word_input.text().strip()
        is_move = self.move_radio.isChecked()
        case_sensitive = self.case_sensitive_cb.isChecked()
        match_exact = self.match_exact_cb.isChecked()
        target_ext = self.ext_combo.currentText()
        target_type = self.target_type_combo.currentText()
        class_remapping = self.class_remap_input.text().strip()
        replace_word_enabled = self.replace_cb.isChecked()
        replace_word = self.replace_input.text()

        if not self.target_dir_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "Please specify the Target Directory.")
            return

        if not img_dir and not mask_dir:
            QMessageBox.warning(self, "Validation Error", "Please specify either an Image Directory or a Mask Directory (or both).")
            return

        # Disable UI during execution
        self.filter_btn.setEnabled(False)
        self.log_browser.clear()
        
        op_str = "Moving" if is_move else "Copying"
        suffix = "all items" if not filter_word else f"items matching '{filter_word}'"
        self._log(f"Starting filter operation: {op_str} {target_type} {suffix}...")
        if case_sensitive:
            self._log("Case sensitive matching enabled.")

        if match_exact:
            self._log("Exact match enabled.")

        # Start worker thread
        self.worker = FilterWorker(img_dir, mask_dir, target_dir, filter_word, is_move, case_sensitive, match_exact, target_ext, target_type, class_remapping, replace_word_enabled, replace_word)
        self.worker.log_msg.connect(self._log)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_progress(self, current, total):
        pass # Optional: Could connect to a QProgressBar here

    def _on_worker_finished(self, success, message):
        """Handle worker completion."""
        self.filter_btn.setEnabled(True)
        if success:
            self._log(f"\nSUCCESS: {message}")
            QMessageBox.information(self, "Operation Complete", message)
        else:
            self._log(f"\nERROR: {message}")
            QMessageBox.critical(self, "Operation Failed", message)

    def get_session_state(self):
        """Return the current paths for session saving."""
        return {
            "img_dir": self.img_dir_input.text(),
            "mask_dir": self.mask_dir_input.text(),
            "mask_enabled": self.mask_enable_cb.isChecked(),
            "target_dir": self.target_dir_input.text(),
            "filter_word": self.filter_word_input.text(),
            "case_sensitive": self.case_sensitive_cb.isChecked(),
            "match_exact": self.match_exact_cb.isChecked(),
            "is_move": self.move_radio.isChecked(),
            "target_ext": self.ext_combo.currentText(),
            "target_type": self.target_type_combo.currentText(),
            "class_remapping": self.class_remap_input.text(),
            "replace_word_enabled": self.replace_cb.isChecked(),
            "replace_word": self.replace_input.text()
        }

    def restore_session_state(self, state):
        """Restore paths and settings from a session dict."""
        if not state:
            return
            
        self.img_dir_input.setText(state.get("img_dir", ""))
        self.mask_dir_input.setText(state.get("mask_dir", ""))
        self.mask_enable_cb.setChecked(state.get("mask_enabled", False))
        self.target_dir_input.setText(state.get("target_dir", ""))
        self.filter_word_input.setText(state.get("filter_word", ""))
        self.case_sensitive_cb.setChecked(state.get("case_sensitive", False))
        self.match_exact_cb.setChecked(state.get("match_exact", False))
        self.target_type_combo.setCurrentText(state.get("target_type", "images"))
        self.class_remap_input.setText(state.get("class_remapping", ""))
        self.replace_cb.setChecked(state.get("replace_word_enabled", False))
        self.replace_input.setText(state.get("replace_word", ""))
        self.ext_combo.setCurrentText(state.get("target_ext", "same as source"))
        
        # Manually trigger type change to set correct enabled states based on restored type
        self._on_target_type_changed(self.target_type_combo.currentText())
        
        if state.get("is_move", False):
            self.move_radio.setChecked(True)
        else:
            self.copy_radio.setChecked(True)
