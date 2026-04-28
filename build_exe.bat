@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onefile ^
  --name StudyPresenceTracker ^
  --add-data "Zinda Bhaag Milkha Bhaag 128 Kbps.mp3;." ^
  --add-data "face_detection_yunet_2023mar.onnx;." ^
  study_presence_tracker.py

if exist "study_tracker.db" copy /Y "study_tracker.db" "dist\study_tracker.db" >nul
echo.
echo Built: dist\StudyPresenceTracker.exe
pause
