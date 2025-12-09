@echo off
echo Building YOLOv8 Annotator executable...
echo.

pyinstaller YOLOv8_Annotator.spec --clean

echo.
echo Build complete! The executable is located in the 'dist' folder.
echo You can find it at: dist\YOLOv8_Annotator.exe
pause
