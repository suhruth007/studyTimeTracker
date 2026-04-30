@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" study_presence_tracker.py
) else (
  python study_presence_tracker.py
)
