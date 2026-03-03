import PyInstaller.__main__
import os

def build():
    # Generate build info
    import datetime
    now = datetime.datetime.now()
    date_str = now.strftime("%Y %b %d")  # e.g., 2026 Jan 14
    
    with open("build_info.py", "w") as f:
        f.write('"""Auto-generated build info"""\n')
        f.write(f'BUILD_DATE = "{date_str}"\n')
    
    print(f"Generated build_info.py with date: {date_str}")

    PyInstaller.__main__.run([
        'main.py',
        '--name=Beaver_yolov8',
        '--onedir',
        '--windowed',
        '--clean',
        '--noconfirm',
        # Collect everything from ultralytics to be safe
        '--collect-all=ultralytics', 
    ])

if __name__ == '__main__':
    build()
