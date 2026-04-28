import datetime as dt
import sqlite3
import sys
import threading
import time
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, Button, Entry, Frame, Label, PhotoImage, StringVar, Text, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk

import cv2
import pyglet

from study_services import StudyStatsService


def app_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(filename):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / filename
    return Path(__file__).resolve().parent / filename


APP_DIR = app_dir()
DB_PATH = APP_DIR / "study_tracker.db"
ALARM_PATH = resource_path("Zinda Bhaag Milkha Bhaag 128 Kbps.mp3")
YUNET_MODEL_PATH = resource_path("face_detection_yunet_2023mar.onnx")
AWAY_THRESHOLD_SECONDS = 120
SNAPSHOT_INTERVAL_SECONDS = 5
PRESENCE_GRACE_SECONDS = 7
PRESENT_HITS_REQUIRED = 1
AWAY_MISSES_REQUIRED = 1
SOFT_FACE_HOLD_SECONDS = 45
CAMERA_INDEX = 0


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def memo_word_count(text):
    return len([word for word in text.strip().split() if word.strip()])


class StudyDatabase:
    def __init__(self, path):
        self.path = path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    active_seconds INTEGER NOT NULL,
                    away_seconds INTEGER NOT NULL,
                    memo TEXT NOT NULL
                )
                """
            )

    def add_session(self, started_at, ended_at, active_seconds, away_seconds, memo):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (started_at, ended_at, active_seconds, away_seconds, memo)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    started_at.isoformat(timespec="seconds"),
                    ended_at.isoformat(timespec="seconds"),
                    int(active_seconds),
                    int(away_seconds),
                    memo.strip(),
                ),
            )

    def daily_totals(self, limit=14):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT substr(started_at, 1, 10) AS day, SUM(active_seconds) AS total
                FROM sessions
                GROUP BY day
                ORDER BY day DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return rows

    def today_total(self):
        today = dt.date.today().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(active_seconds), 0)
                FROM sessions
                WHERE substr(started_at, 1, 10) = ?
                """,
                (today,),
            ).fetchone()
        return int(row[0] or 0)


class AlarmPlayer:
    def __init__(self, alarm_path):
        self.alarm_path = alarm_path
        self.ready = self.alarm_path.exists()
        self.playing = False
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.error = ""
        if not self.alarm_path.exists():
            self.error = "Alarm file does not exist"

    def start(self):
        with self.lock:
            if not self.ready or self.playing:
                return
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._play_loop, daemon=True)
            self.thread.start()
            self.playing = True

    def _play_loop(self):
        player = None
        try:
            source = pyglet.media.load(str(self.alarm_path), streaming=False)
            player = pyglet.media.Player()
            player.queue(source)
            player.loop = True
            player.play()
            while not self.stop_event.is_set():
                pyglet.clock.tick()
                time.sleep(0.05)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if player is not None:
                player.pause()
                player.delete()
            with self.lock:
                self.playing = False

    def stop(self):
        self.stop_event.set()


class CameraPresence:
    def __init__(self, on_status):
        self.on_status = on_status
        self.running = False
        self.present = False
        self.last_seen = 0
        self.error = ""
        self.thread = None
        self.detector_name = "YuNet"
        self.yunet = None
        if YUNET_MODEL_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
            self.yunet = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL_PATH),
                "",
                (320, 320),
                score_threshold=0.78,
                nms_threshold=0.3,
                top_k=20,
            )
        else:
            self.detector_name = "Haar fallback"
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        eye_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
        upper_body_path = Path(cv2.data.haarcascades) / "haarcascade_upperbody.xml"
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))
        self.eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))
        self.upper_body_cascade = cv2.CascadeClassifier(str(upper_body_path))
        self.hit_count = 0
        self.miss_count = 0
        self.last_confirmed = 0
        self.last_status = ""
        self.preview_lock = threading.Lock()
        self.preview_png = None
        self.preview_signal = "none"
        self.last_checked_at = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def is_present(self):
        if self.last_seen == 0:
            return False
        return time.monotonic() - self.last_seen <= PRESENCE_GRACE_SECONDS

    def mark_present_manual(self):
        now = time.monotonic()
        self.present = True
        self.last_seen = now
        self.last_confirmed = now
        self.hit_count = PRESENT_HITS_REQUIRED
        self.miss_count = 0
        self._set_status("Present (manual)")

    def _loop(self):
        capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(CAMERA_INDEX)

        if not capture.isOpened():
            self.error = "Camera not available"
            self.on_status(self.error)
            self.running = False
            return

        self.on_status(f"Camera ready ({self.detector_name}, {SNAPSHOT_INTERVAL_SECONDS}s snapshots)")
        while self.running:
            ok, frame = capture.read()
            if not ok:
                self.error = "Unable to read camera"
                self.on_status(self.error)
                time.sleep(1)
                continue

            self.last_checked_at = time.monotonic()
            signal, face_boxes, eye_boxes = self._detect_presence_signal(frame)

            if signal in ("confirmed", "body"):
                self.hit_count += 1
                self.miss_count = 0
                if signal == "confirmed":
                    self.last_confirmed = time.monotonic()
            elif signal == "soft" and self.present and time.monotonic() - self.last_confirmed <= SOFT_FACE_HOLD_SECONDS:
                self.hit_count = max(self.hit_count, PRESENT_HITS_REQUIRED)
                self.miss_count = 0
            else:
                self.miss_count += 1
                self.hit_count = 0

            if self.hit_count >= PRESENT_HITS_REQUIRED:
                self.last_seen = time.monotonic()
                if not self.present:
                    self.present = True
                    self._set_status("Present")
            elif self.present and self.miss_count >= AWAY_MISSES_REQUIRED and not self.is_present():
                self.present = False
                self._set_status("Away")

            self._store_preview(frame, signal, face_boxes, eye_boxes)
            time.sleep(SNAPSHOT_INTERVAL_SECONDS)

        capture.release()

    def get_preview(self):
        with self.preview_lock:
            return self.preview_png, self.preview_signal

    def _set_status(self, text):
        if text != self.last_status:
            self.last_status = text
            self.on_status(text)

    def _detect_presence_signal(self, frame):
        if self.yunet is not None:
            signal, face_boxes, eye_boxes = self._detect_with_yunet(frame)
        else:
            signal, face_boxes, eye_boxes = self._detect_with_haar(frame)

        if signal == "confirmed":
            return signal, face_boxes, eye_boxes

        body_signal, body_boxes = self._detect_upper_body(frame)
        if body_signal == "body":
            return "body", face_boxes + body_boxes, eye_boxes
        return signal, face_boxes, eye_boxes

    def _detect_upper_body(self, frame):
        if self.upper_body_cascade.empty():
            return "none", []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        height, width = gray.shape[:2]
        bodies = self.upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(max(100, width // 5), max(90, height // 5)),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        boxes = []
        for (x, y, w, h) in bodies:
            if w / max(h, 1) < 0.65 or w / max(h, 1) > 2.2:
                continue
            if y > height * 0.85:
                continue
            boxes.append((x, y, w, h))
        return ("body" if boxes else "none"), boxes

    def _detect_with_yunet(self, frame):
        height, width = frame.shape[:2]
        self.yunet.setInputSize((width, height))
        _retval, faces = self.yunet.detect(frame)
        if faces is None or len(faces) == 0:
            return "none", [], []

        face_boxes = []
        eye_boxes = []
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            score = float(face[-1])
            if score < 0.78:
                continue
            if w < max(70, width // 14) or h < max(70, height // 14):
                continue
            if w / max(h, 1) < 0.55 or w / max(h, 1) > 1.55:
                continue
            if y < -10 or y + h > height + 10:
                continue

            face_boxes.append((x, y, w, h))
            landmarks = face[4:14].reshape((5, 2)).astype(int)
            left_eye, right_eye = landmarks[0], landmarks[1]
            eye_distance = abs(int(left_eye[0]) - int(right_eye[0]))
            if eye_distance >= max(18, w * 0.18):
                eye_size = max(10, int(w * 0.08))
                for ex, ey in (left_eye, right_eye):
                    eye_boxes.append((int(ex - eye_size / 2), int(ey - eye_size / 2), eye_size, eye_size))
                return "confirmed", face_boxes, eye_boxes

        return ("soft" if face_boxes else "none"), face_boxes, eye_boxes

    def _detect_with_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        height, width = gray.shape[:2]
        min_face = max(90, width // 10)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=7,
            minSize=(min_face, min_face),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return "none", [], []

        soft_face_found = False
        face_boxes = []
        eye_boxes = []
        for (x, y, w, h) in faces:
            if w / max(h, 1) < 0.65 or w / max(h, 1) > 1.45:
                continue
            if y < height * 0.03 or y + h > height * 0.95:
                continue
            soft_face_found = True
            face_boxes.append((x, y, w, h))

            upper_face = gray[y : y + int(h * 0.65), x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(18, 18),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(eyes) >= 1:
                eye_boxes.extend([(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes])
                return "confirmed", face_boxes, eye_boxes

        # Looking down at a book often hides the eyes. This signal is allowed
        # only after a real face has already been confirmed recently.
        return ("soft" if soft_face_found else "none"), face_boxes, eye_boxes

    def _store_preview(self, frame, signal, face_boxes, eye_boxes):
        preview = frame.copy()
        color = {
            "confirmed": (34, 197, 94),
            "body": (168, 85, 247),
            "soft": (0, 215, 255),
            "none": (120, 120, 120),
        }.get(signal, (120, 120, 120))

        for (x, y, w, h) in face_boxes:
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        for (x, y, w, h) in eye_boxes:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (56, 189, 248), 1)

        label = f"{self.detector_name} snapshot: {signal.upper()} | misses {self.miss_count}"
        cv2.rectangle(preview, (10, 10), (10 + min(430, len(label) * 9), 42), (11, 18, 32), -1)
        cv2.putText(preview, label, (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

        preview = cv2.resize(preview, (360, 270), interpolation=cv2.INTER_AREA)
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", preview)
        if ok:
            with self.preview_lock:
                self.preview_png = encoded.tobytes()
                self.preview_signal = signal


class StudyTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Study Presence Tracker")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)

        self.db = StudyDatabase(DB_PATH)
        self.stats = StudyStatsService(DB_PATH)
        self.alarm = AlarmPlayer(ALARM_PATH)
        self.camera = CameraPresence(self.set_camera_status)

        self.session_running = False
        self.session_started_at = None
        self.active_seconds = 0
        self.away_seconds = 0
        self.last_tick = None
        self.away_started_monotonic = None
        self.paused_by_away = False
        self.testing_alarm = False

        self.timer_var = StringVar(value="00:00:00")
        self.today_var = StringVar(value="Today: 00:00:00")
        self.goal_var = StringVar(value="Goal: --")
        self.streak_var = StringVar(value="Streak: 0 days")
        self.goal_entry_var = StringVar(value="")
        self.status_var = StringVar(value="Ready")
        self.camera_var = StringVar(value="Camera starting...")
        self.away_var = StringVar(value="Away: 00:00")
        self.preview_status_var = StringVar(value="Camera Preview")
        self.preview_image = None
        self.widget_window = None
        self.widget_drag_offset = (0, 0)

        self._build_ui()
        self.create_corner_widget()
        self.load_goal_display()
        self.camera.start()
        self.refresh_history()
        self.root.after(1000, self.tick)
        self.root.after(150, self.update_camera_preview)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if not ALARM_PATH.exists():
            messagebox.showwarning("Alarm missing", f"Alarm file was not found:\n{ALARM_PATH}")
        elif not self.alarm.ready:
            messagebox.showwarning("Alarm unavailable", f"Could not load alarm audio:\n{getattr(self.alarm, 'error', '')}")

    def create_corner_widget(self):
        widget = Toplevel(self.root)
        widget.title("Study Widget")
        widget.geometry("260x136+0+0")
        widget.resizable(False, False)
        widget.configure(bg="#020617")
        widget.attributes("-topmost", True)
        widget.overrideredirect(True)
        self.widget_window = widget

        outer = Frame(widget, bg="#020617", padx=10, pady=8, highlightbackground="#38bdf8", highlightthickness=1)
        outer.pack(fill=BOTH, expand=True)
        outer.bind("<ButtonPress-1>", self.start_widget_drag)
        outer.bind("<B1-Motion>", self.drag_widget)

        top_row = Frame(outer, bg="#020617")
        top_row.pack(fill=BOTH)
        top_row.bind("<ButtonPress-1>", self.start_widget_drag)
        top_row.bind("<B1-Motion>", self.drag_widget)

        Label(
            top_row,
            textvariable=self.timer_var,
            bg="#020617",
            fg="#38bdf8",
            font=("Consolas", 20, "bold"),
        ).pack(side=LEFT)

        Button(
            top_row,
            text="x",
            command=widget.withdraw,
            bg="#111827",
            fg="#e5e7eb",
            activebackground="#1f2937",
            activeforeground="#ffffff",
            width=3,
            relief="flat",
        ).pack(side=RIGHT)

        Label(
            outer,
            textvariable=self.status_var,
            bg="#020617",
            fg="#facc15",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill=BOTH, pady=(4, 0))

        Label(
            outer,
            textvariable=self.away_var,
            bg="#020617",
            fg="#fca5a5",
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill=BOTH)

        buttons = Frame(outer, bg="#020617")
        buttons.pack(fill=BOTH, pady=(6, 0))

        Button(
            buttons,
            text="Start",
            command=self.start_session,
            bg="#0ea5e9",
            fg="#ffffff",
            activebackground="#0284c7",
            activeforeground="#ffffff",
            width=7,
            relief="flat",
        ).pack(side=LEFT, padx=(0, 5))

        Button(
            buttons,
            text="Stop",
            command=self.request_stop_session,
            bg="#ef4444",
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            width=7,
            relief="flat",
        ).pack(side=LEFT, padx=(0, 5))

        Button(
            buttons,
            text="Here",
            command=self.mark_manual_here,
            bg="#22c55e",
            fg="#ffffff",
            activebackground="#16a34a",
            activeforeground="#ffffff",
            width=7,
            relief="flat",
        ).pack(side=LEFT)

        Button(
            outer,
            text="Dashboard",
            command=self.show_dashboard,
            bg="#1f2937",
            fg="#ffffff",
            activebackground="#374151",
            activeforeground="#ffffff",
            relief="flat",
        ).pack(fill=BOTH, pady=(6, 0))

    def start_widget_drag(self, event):
        self.widget_drag_offset = (event.x_root - self.widget_window.winfo_x(), event.y_root - self.widget_window.winfo_y())

    def drag_widget(self, event):
        x_offset, y_offset = self.widget_drag_offset
        x = max(0, event.x_root - x_offset)
        y = max(0, event.y_root - y_offset)
        self.widget_window.geometry(f"+{x}+{y}")

    def show_dashboard(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def show_corner_widget(self):
        if self.widget_window:
            self.widget_window.deiconify()
            self.widget_window.attributes("-topmost", True)
            self.widget_window.lift()

    def _build_ui(self):
        self.root.configure(bg="#0b1220")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#111827", foreground="#e5e7eb", fieldbackground="#111827", rowheight=30)
        style.configure("Treeview.Heading", background="#1f2937", foreground="#f9fafb", font=("Segoe UI", 10, "bold"))

        main = Frame(self.root, bg="#0b1220", padx=24, pady=22)
        main.pack(fill=BOTH, expand=True)

        top = Frame(main, bg="#0b1220")
        top.pack(fill=BOTH, expand=False)

        title = Label(top, text="Study Presence Tracker", bg="#0b1220", fg="#f9fafb", font=("Segoe UI", 22, "bold"))
        title.pack(anchor="w")

        subtitle = Label(
            top,
            text="Camera checks one snapshot every 5 seconds. Alarm starts after 2 minutes away.",
            bg="#0b1220",
            fg="#9ca3af",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor="w", pady=(4, 18))

        dashboard = Frame(main, bg="#111827", padx=22, pady=20, highlightbackground="#263244", highlightthickness=1)
        dashboard.pack(fill=BOTH, expand=False)

        left_panel = Frame(dashboard, bg="#111827")
        left_panel.pack(side=LEFT, fill=BOTH, expand=True)

        timer = Label(left_panel, textvariable=self.timer_var, bg="#111827", fg="#38bdf8", font=("Consolas", 48, "bold"))
        timer.pack(anchor="w")

        today = Label(left_panel, textvariable=self.today_var, bg="#111827", fg="#d1d5db", font=("Segoe UI", 14, "bold"))
        today.pack(anchor="w", pady=(4, 2))

        goal = Label(left_panel, textvariable=self.goal_var, bg="#111827", fg="#d1d5db", font=("Segoe UI", 11, "bold"))
        goal.pack(anchor="w", pady=(2, 0))

        streak = Label(left_panel, textvariable=self.streak_var, bg="#111827", fg="#a7f3d0", font=("Segoe UI", 11, "bold"))
        streak.pack(anchor="w", pady=(2, 0))

        status = Label(left_panel, textvariable=self.status_var, bg="#111827", fg="#facc15", font=("Segoe UI", 12))
        status.pack(anchor="w", pady=(8, 0))

        camera = Label(left_panel, textvariable=self.camera_var, bg="#111827", fg="#a7f3d0", font=("Segoe UI", 11))
        camera.pack(anchor="w", pady=(4, 0))

        away = Label(left_panel, textvariable=self.away_var, bg="#111827", fg="#fca5a5", font=("Segoe UI", 11))
        away.pack(anchor="w", pady=(4, 0))

        controls = Frame(dashboard, bg="#111827")
        controls.pack(side=RIGHT, fill=BOTH, padx=(18, 0))

        self.start_button = Button(
            controls,
            text="Start",
            command=self.start_session,
            bg="#0ea5e9",
            fg="#ffffff",
            activebackground="#0284c7",
            activeforeground="#ffffff",
            font=("Segoe UI", 13, "bold"),
            width=14,
            height=2,
            relief="flat",
        )
        self.start_button.pack(side=TOP, pady=(6, 12))

        self.stop_button = Button(
            controls,
            text="Stop",
            command=self.request_stop_session,
            bg="#ef4444",
            fg="#ffffff",
            activebackground="#dc2626",
            activeforeground="#ffffff",
            font=("Segoe UI", 13, "bold"),
            width=14,
            height=2,
            relief="flat",
            state="disabled",
        )
        self.stop_button.pack(side=TOP)

        self.manual_here_button = Button(
            controls,
            text="I'm Here",
            command=self.mark_manual_here,
            bg="#22c55e",
            fg="#ffffff",
            activebackground="#16a34a",
            activeforeground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            width=14,
            height=2,
            relief="flat",
        )
        self.manual_here_button.pack(side=TOP, pady=(12, 0))

        self.test_alarm_button = Button(
            controls,
            text="Test Alarm",
            command=self.test_alarm,
            bg="#374151",
            fg="#ffffff",
            activebackground="#4b5563",
            activeforeground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            width=14,
            height=2,
            relief="flat",
        )
        self.test_alarm_button.pack(side=TOP, pady=(12, 0))

        Button(
            controls,
            text="Show Widget",
            command=self.show_corner_widget,
            bg="#374151",
            fg="#ffffff",
            activebackground="#4b5563",
            activeforeground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            width=14,
            height=2,
            relief="flat",
        ).pack(side=TOP, pady=(12, 0))

        goal_box = Frame(controls, bg="#111827")
        goal_box.pack(side=TOP, fill=BOTH, pady=(12, 0))

        self.goal_entry = Entry(
            goal_box,
            textvariable=self.goal_entry_var,
            bg="#020617",
            fg="#f9fafb",
            insertbackground="#f9fafb",
            relief="flat",
            width=8,
            font=("Segoe UI", 11),
        )
        self.goal_entry.pack(side=LEFT, fill=BOTH, expand=True)

        Button(
            goal_box,
            text="Set Goal",
            command=self.save_daily_goal,
            bg="#374151",
            fg="#ffffff",
            activebackground="#4b5563",
            activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
        ).pack(side=RIGHT, padx=(8, 0))

        Button(
            controls,
            text="Export CSV",
            command=self.export_csv,
            bg="#374151",
            fg="#ffffff",
            activebackground="#4b5563",
            activeforeground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            width=14,
            height=2,
            relief="flat",
        ).pack(side=TOP, pady=(12, 0))

        preview_title = Label(
            controls,
            textvariable=self.preview_status_var,
            bg="#111827",
            fg="#f9fafb",
            font=("Segoe UI", 11, "bold"),
        )
        preview_title.pack(side=TOP, anchor="w", pady=(18, 8))

        preview_box = Frame(controls, bg="#020617", width=360, height=270)
        preview_box.pack(side=TOP)
        preview_box.pack_propagate(False)

        self.preview_label = Label(
            preview_box,
            text="Waiting for camera...",
            bg="#020617",
            fg="#9ca3af",
            relief="flat",
        )
        self.preview_label.pack(fill=BOTH, expand=True)

        history_label = Label(main, text="Daily Study History", bg="#0b1220", fg="#f9fafb", font=("Segoe UI", 15, "bold"))
        history_label.pack(anchor="w", pady=(24, 10))

        self.history = ttk.Treeview(main, columns=("day", "total", "goal"), show="headings", height=10)
        self.history.heading("day", text="Date")
        self.history.heading("total", text="Study time")
        self.history.heading("goal", text="Goal")
        self.history.column("day", anchor="w", width=180)
        self.history.column("total", anchor="w", width=180)
        self.history.column("goal", anchor="w", width=180)
        self.history.pack(fill=BOTH, expand=True)

    def set_camera_status(self, text):
        self.root.after(0, lambda: self.camera_var.set(f"Camera: {text}"))

    def update_camera_preview(self):
        preview_png, signal = self.camera.get_preview()
        if preview_png:
            self.preview_image = PhotoImage(data=preview_png)
            self.preview_label.configure(image=self.preview_image, text="")
            self.preview_status_var.set(f"Camera Preview: {signal.title()}")
        self.root.after(150, self.update_camera_preview)

    def load_goal_display(self):
        goal_seconds = self.stats.daily_goal_seconds()
        self.goal_entry_var.set(str(round(goal_seconds / 3600, 2)))
        self.update_stats_display()

    def update_stats_display(self):
        goal_seconds = self.stats.daily_goal_seconds()
        today_total = self.stats.today_total()
        if self.session_running:
            today_total += int(self.active_seconds)
        self.goal_var.set(f"Goal: {format_duration(today_total)} / {format_duration(goal_seconds)}")
        streak = self.stats.current_streak_days()
        self.streak_var.set(f"Streak: {streak} day{'s' if streak != 1 else ''}")

    def save_daily_goal(self):
        try:
            self.stats.set_daily_goal_hours(self.goal_entry_var.get())
        except ValueError as exc:
            messagebox.showwarning("Invalid goal", str(exc))
            return
        self.update_stats_display()
        self.refresh_history()
        self.status_var.set("Daily goal saved")

    def export_csv(self):
        default_name = f"study_sessions_{dt.date.today().isoformat()}.csv"
        output_path = filedialog.asksaveasfilename(
            title="Export study sessions",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not output_path:
            return
        saved_path = self.stats.export_sessions_csv(output_path)
        self.status_var.set(f"Exported CSV: {saved_path.name}")

    def start_session(self):
        self.session_running = True
        self.session_started_at = dt.datetime.now()
        self.active_seconds = 0
        self.away_seconds = 0
        self.last_tick = time.monotonic()
        self.away_started_monotonic = None
        self.paused_by_away = False
        self.status_var.set("Session running")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def mark_manual_here(self):
        self.camera.mark_present_manual()
        self.away_started_monotonic = None
        self.paused_by_away = False
        self.away_var.set("Away: 00:00")
        self.alarm.stop()
        self.status_var.set("Manual presence marked")

    def request_stop_session(self):
        if not self.session_running:
            return
        self.open_memo_dialog()

    def test_alarm(self):
        if not self.alarm.ready:
            messagebox.showwarning("Alarm unavailable", self.alarm.error or "Alarm file is not ready.")
            return
        self.testing_alarm = True
        self.alarm.start()
        self.status_var.set("Testing alarm...")
        self.root.after(4000, self.stop_alarm_test)

    def stop_alarm_test(self):
        if self.testing_alarm:
            self.alarm.stop()
            self.testing_alarm = False
            if not self.session_running:
                self.status_var.set("Ready")

    def open_memo_dialog(self):
        dialog = Toplevel(self.root)
        dialog.title("Session memo")
        dialog.geometry("560x360")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg="#0b1220")

        Label(
            dialog,
            text="What did you study or work on?",
            bg="#0b1220",
            fg="#f9fafb",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 4))

        Label(
            dialog,
            text="Enter at least 10 words before stopping the session.",
            bg="#0b1220",
            fg="#9ca3af",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=18, pady=(0, 12))

        memo_box = Text(dialog, height=9, wrap="word", bg="#111827", fg="#f9fafb", insertbackground="#f9fafb", relief="flat")
        memo_box.pack(fill=BOTH, expand=True, padx=18)
        feedback = StringVar(value="Words: 0 / 10")
        Label(dialog, textvariable=feedback, bg="#0b1220", fg="#fca5a5", font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=(8, 0))

        buttons = Frame(dialog, bg="#0b1220")
        buttons.pack(fill=BOTH, padx=18, pady=16)

        def update_feedback(_event=None):
            count = memo_word_count(memo_box.get("1.0", END))
            feedback.set(f"Words: {count} / 10")

        def save_and_close():
            memo = memo_box.get("1.0", END).strip()
            count = memo_word_count(memo)
            if count < 10:
                feedback.set(f"Words: {count} / 10 - add more detail")
                return
            self.stop_session(memo)
            dialog.destroy()

        memo_box.bind("<KeyRelease>", update_feedback)
        Button(buttons, text="Cancel", command=dialog.destroy, width=12, bg="#374151", fg="#ffffff", relief="flat").pack(side=RIGHT, padx=(8, 0))
        Button(buttons, text="Save Session", command=save_and_close, width=14, bg="#22c55e", fg="#ffffff", relief="flat").pack(side=RIGHT)
        memo_box.focus_set()

    def stop_session(self, memo):
        ended_at = dt.datetime.now()
        self.session_running = False
        self.alarm.stop()
        self.db.add_session(self.session_started_at, ended_at, self.active_seconds, self.away_seconds, memo)
        self.status_var.set("Session saved")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.refresh_history()

    def tick(self):
        now = time.monotonic()
        if self.session_running and self.last_tick is not None:
            elapsed = max(0, now - self.last_tick)
            present = self.camera.is_present()

            if present:
                self.away_started_monotonic = None
                self.paused_by_away = False
                if not self.testing_alarm:
                    self.alarm.stop()
                self.active_seconds += elapsed
                self.status_var.set("Session running")
            else:
                if self.away_started_monotonic is None:
                    self.away_started_monotonic = now
                away_elapsed = now - self.away_started_monotonic
                self.away_seconds += elapsed
                self.away_var.set(f"Away: {format_duration(away_elapsed)[3:]}")
                if away_elapsed >= AWAY_THRESHOLD_SECONDS:
                    self.paused_by_away = True
                    self.alarm.start()
                    self.status_var.set("Paused: away for 2 minutes. Alarm active.")
                else:
                    self.status_var.set(f"Away grace: {format_duration(AWAY_THRESHOLD_SECONDS - away_elapsed)[3:]} left")

            self.timer_var.set(format_duration(self.active_seconds))
            today_total = self.stats.today_total() + int(self.active_seconds)
            self.today_var.set(f"Today: {format_duration(today_total)}")
            self.update_stats_display()
            if present:
                self.away_var.set("Away: 00:00")

        elif not self.session_running:
            self.today_var.set(f"Today: {format_duration(self.stats.today_total())}")
            self.update_stats_display()

        self.last_tick = now
        self.root.after(1000, self.tick)

    def refresh_history(self):
        for item in self.history.get_children():
            self.history.delete(item)
        today = dt.date.today().isoformat()
        totals = dict(self.stats.daily_totals(limit=14))
        if today not in totals:
            totals[today] = 0
        goal_seconds = self.stats.daily_goal_seconds()
        for day in sorted(totals.keys(), reverse=True):
            goal_state = "Met" if totals[day] >= goal_seconds else "Not met"
            self.history.insert("", END, values=(day, format_duration(totals[day]), goal_state))

    def on_close(self):
        if self.session_running:
            if not messagebox.askyesno("Session running", "A session is running. Close without saving it?"):
                return
        self.alarm.stop()
        self.camera.stop()
        if self.widget_window:
            self.widget_window.destroy()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = StudyTrackerApp(root)
    root.mainloop()
