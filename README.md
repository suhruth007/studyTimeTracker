# Study Presence Tracker

A Windows desktop study timer that uses your webcam to detect whether you are present.

## Setup

Install dependencies:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
s
Run the app:

```powershell
.venv\Scripts\python.exe study_presence_tracker.py
```

Or double-click:

```text
run_study_tracker.bat
```

Build the no-install Windows exe:

```text
build_exe.bat
```

The portable exe is created at `dist\StudyPresenceTracker.exe`. Keep `study_tracker.db` beside it if you want to preserve the same history.

## Behavior

- Press **Start** to begin a study session.
- The camera checks one snapshot every 5 seconds using OpenCV YuNet face detection plus an upper-body fallback.
- The timer counts while your recent presence snapshot is valid.
- If you leave the camera view, the app gives a 2 minute grace period.
- After 2 minutes away, the timer pauses and the alarm plays.
- The alarm stops automatically when your face is detected again.
- Use **Test Alarm** to check MP3 playback without waiting for the away timer.
- The camera preview shows detection boxes for testing:
  - green: confirmed face
  - purple: upper-body signal
  - yellow: soft face signal
  - gray: no usable face signal
- Use **I'm Here** to manually reset away status if detection misses you.
- Set a daily goal in hours, track your streak, and export session logs to CSV.
- A compact always-on-top widget appears in the top-left corner with timer, status, Start, Stop, Here, and Dashboard controls.
- Press **Stop** to save the session.
- A memo with at least 10 words is required before the session is saved.
- Study history is stored locally in `study_tracker.db`.

## Files

- `study_presence_tracker.py` - desktop app
- `study_services.py` - daily goal, streak, and CSV export service
- `build_exe.bat` - creates the portable Windows exe
- `study_tracker.db` - created automatically after first save
- `face_detection_yunet_2023mar.onnx` - OpenCV face detection model
- `Zinda Bhaag Milkha Bhaag 128 Kbps.mp3` - alarm sound
- `.venv` - local Python environment created for this app
