# YOLOv8 Annotator

A desktop application for annotating images with polygon annotations in YOLOv8 format.

## Features

- Load and display images from a folder
- View existing YOLO format annotations
- Add, edit, and delete polygon annotations
- Toggle annotation visibility
- Navigate between images with keyboard shortcuts
- Auto-save functionality
- Class management
- Support for multiple image formats (JPG, PNG, BMP, TIFF)

## Installation

### 1. Activate the conda environment

```bash
conda activate yolov8_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PySide6 (Qt GUI framework)
- Pillow (Image processing)
- numpy (Numerical operations)

## Usage

### Starting the Application

```bash
python main.py
```

### Workflow

1. **Select Folders**
   - Click "Select Images Folder" and choose the folder containing your images
   - Click "Select Labels Folder" and choose the folder for annotation files
   - The first image will load automatically

2. **View Annotations**
   - Existing annotations are displayed as colored polygons
   - Each class has a different color
   - Press `Space` to toggle annotation visibility

3. **Add Annotations**
   - Select the class from the dropdown
   - Click "Add Polygon" button
   - Click on the image to place polygon vertices
   - Double-click or press `Enter` to finish the polygon
   - Press `Escape` to cancel

4. **Edit Annotations**
   - Click inside a polygon to select it
   - Selected polygons are highlighted in yellow
   - Drag vertices to adjust the shape
   - Press `Delete` to remove the selected annotation

5. **Navigate Images**
   - Press `→` (Right Arrow) for next image
   - Press `←` (Left Arrow) for previous image
   - Or use the "Next" and "Previous" buttons

6. **Save Annotations**
   - Press `Ctrl+S` to save current annotations
   - Or click the "Save" button
   - Annotations are auto-saved when navigating to another image

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` | Next image |
| `←` | Previous image |
| `Space` | Toggle annotation visibility |
| `Ctrl+S` | Save annotations |
| `Delete` | Delete selected annotation |
| `Enter` | Finish drawing polygon |
| `Escape` | Cancel current operation |

## YOLO Format

Annotations are saved in YOLO v8 polygon format:

```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

Where:
- `class_id`: Integer class identifier (0, 1, 2, ...)
- `x, y`: Normalized coordinates (0.0 to 1.0) relative to image dimensions

### Example Annotation File (image001.txt)

```
0 0.5 0.3 0.6 0.3 0.6 0.7 0.5 0.7
1 0.2 0.2 0.3 0.2 0.3 0.4 0.2 0.4
```

### Class Names

Create a `classes.txt` file in your labels folder with one class name per line:

```
person
car
bicycle
```

## Project Structure

```
yolov8_annotator/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── ui/
│   └── main_window.py      # Main application window
├── widgets/
│   ├── image_canvas.py     # Image display and annotation widget
│   └── annotation_list.py  # Annotation list widget
├── utils/
│   ├── yolo_format.py      # YOLO format utilities
│   └── file_handler.py     # File management utilities
└── models/
    └── annotation.py       # Annotation data model
```

## Tips

- **Auto-save**: Annotations are automatically saved when switching images
- **File Creation**: Annotation files are created automatically when you add annotations
- **File Deletion**: Empty annotation files are automatically deleted
- **Multiple Classes**: Add more classes by editing `classes.txt` in your labels folder
- **Zoom**: The image automatically fits the window; resize the window to see more detail

## Troubleshooting

### Import errors
Make sure you've activated the conda environment and installed all dependencies:
```bash
conda activate yolov8_env
pip install -r requirements.txt
```

### Images not loading
- Check that the images folder contains supported formats (JPG, PNG, BMP, TIFF)
- Verify folder permissions

### Annotations not saving
- Ensure the labels folder exists and is writable
- Check that you have proper file permissions

## Requirements

- Python 3.8+
- PySide6 >= 6.5.0
- Pillow >= 10.0.0
- numpy >= 1.24.0

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
