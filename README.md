# YOLOv8 Annotator

A desktop application for annotating images with polygon annotations in YOLOv8 format, and performing video inference with YOLO models.

## Features

### Image Annotation
- Load and display images from a folder
- View existing YOLO format annotations
- Add, edit, and delete polygon annotations
- **Resizable Panels**: Customize the layout by dragging panel borders
- Toggle annotation visibility
- Navigate between images with keyboard shortcuts
- Auto-save functionality
- Class management
- Support for multiple image formats (JPG, PNG, BMP, TIFF)

### Video Inference
- **Dual Model Support**: Load two different YOLO models and switch between them instantly
- **Video Player**: Play, pause, stop, step forward/backward, and seek through videos
- **Real-time Inference**: Toggle inference on/off to see model detections overlaid on the video
- **Export**: Export specific frames with their detections (images + label files) to a dataset
- **Resizable Panels**: Adjust the video list and player size

### Session Management
- **Save/Load Sessions**: Save your current work state (opened folders, loaded models, current image/video, layout) and resume later
- **Auto-load**: Automatically loads the last session on startup

## Installation

### 1. Create/Update the conda environment

```bash
conda env create -f environment.yml
```

Or if updating an existing environment:

```bash
conda env update -f environment.yml --prune
```

### 2. Activate the environment

```bash
conda activate yolov8_annotator_env
```

This will install all dependencies including:
- PySide6 (Qt GUI framework)
- Pillow (Image processing)
- numpy (Numerical operations)
- opencv-python (Video processing)
- ultralytics (YOLOv8 inference)
- torch, torchvision (Deep learning backend)

## Usage

### Starting the Application

```bash
python main.py
```

### Image Annotation Workflow

1. **Select Folders**
   - Click "Select Images Folder" and choose the folder containing your images
   - Click "Select Labels Folder" and choose the folder for annotation files
   - The first image will load automatically

2. **Add Annotations**
   - Select the class from the dropdown
   - Click "Add Polygon" button
   - **Hold `Shift` + Click** to place polygon vertices (This allows you to pan/zoom between points if needed, though pan/zoom is not fully implemented yet, the shift modifier prevents accidental clicks)
   - Double-click or press `Enter` to finish the polygon
   - Press `Escape` to cancel

3. **Edit Annotations**
   - Click inside a polygon to select it (Highlighted in yellow)
   - Drag vertices to adjust the shape
   - Press `Delete` to remove the selected annotation
   - Change the class of a selected annotation using the dropdown sidebar

4. **Navigate Images**
   - Press `→` (Right Arrow) or `←` (Left Arrow)
   - Or use the "Next" and "Previous" buttons

5. **Save Annotations**
   - Press `Ctrl+S` or click "Save"
   - Annotations are auto-saved when navigating to another image

### Video Inference Workflow

1. **Select Video Folder**
   - Switch to the "Video Inference" tab
   - Click "Select Folder" to load videos from a directory

2. **Load Models**
   - You can load up to two YOLO`.pt` models
   - Click "Load Model 1" or "Load Model 2"
   - Switch between them using the radio buttons

3. **Inference & Playback**
   - Select a video from the list
   - Use playback controls (Play, Pause, Step) to navigate
   - Ensure "Enable Inference" is checked to see real-time detections
   - Adjust "Confidence Threshold" slider to filter low-confidence detections

4. **Export Data**
   - Select an "Export Output Folder"
   - Pause the video at a frame you want to keep
   - Click "Export Frame and Annotation"
   - The frame will be saved as an image, and detections as a text file in YOLO format in the output directory

## Keyboard Shortcuts

| Key | Action | Context |
|-----|--------|---------|
| `→` | Next image | Annotation Tab |
| `←` | Previous image | Annotation Tab |
| `Space` | Toggle annotation visibility | Annotation Tab |
| `Ctrl+S` | Save annotations | Annotation Tab |
| `Delete` | Delete selected annotation | Annotation Tab |
| `Shift` + Click | Add polygon vertex | Annotation Tab (Drawing Mode) |
| `Enter` | Finish drawing polygon | Annotation Tab (Drawing Mode) |
| `Escape` | Cancel operation / Deselect | Global |
| `Ctrl+Z` | Undo | Annotation Tab |
| `Ctrl+Y` | Redo | Annotation Tab |

## Project Structure

```
yolov8_annotator/
├── main.py                     # Application entry point
├── environment.yml             # Conda environment definition
├── ui/
│   └── main_window.py          # Main application window
├── widgets/
│   ├── image_canvas.py         # Image display and annotation widget
│   ├── annotation_list.py      # Annotation list widget
│   ├── image_list.py           # Image list widget
│   ├── video_inference_tab.py  # Video inference main tab
│   ├── video_player.py         # Video player widget
│   └── video_list.py           # Video file list widget
├── utils/
│   ├── yolo_format.py          # YOLO format utilities
│   ├── file_handler.py         # File management utilities
│   ├── video_handler.py        # Video file management
│   ├── video_thread.py         # Threading for video playback
│   ├── yolo_inference.py       # YOLO inference engine
│   └── session_manager.py      # Session state management
└── models/
    └── annotation.py           # Annotation data model
```

## Tips

- **Resizable Panels**: You can drag the borders between the file lists and the main view to adjust their size. This is useful for seeing long filenames.
- **Session**: The app remembers your last session. If you close it and reopen it, it will try to take you back to where you left off.
- **Exporting**: Use the "Copy" button next to the export path to quickly get the directory path for checking your exported datasets.

## Building the Executable

To create a standalone executable:

1.  **Activate Environment**
    ```bash
    conda activate yolov8_annotator_env
    pip install pyinstaller
    ```

2.  **Run Build Script**
    ```bash
    python build_app.py
    ```

3.  **Locate EXE**
    The app will be built in `dist/YOLOv8Annotator/YOLOv8Annotator.exe`.

## License

This project is open source and available for educational and commercial use.
