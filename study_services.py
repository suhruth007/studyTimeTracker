import csv
import datetime as dt
import sqlite3
from pathlib import Path


DEFAULT_DAILY_GOAL_SECONDS = 4 * 3600


class StudyStatsService:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._init_settings()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_settings(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO settings (key, value)
                VALUES ('daily_goal_seconds', ?)
                """,
                (str(DEFAULT_DAILY_GOAL_SECONDS),),
            )

    def daily_goal_seconds(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = 'daily_goal_seconds'"
            ).fetchone()
        if not row:
            return DEFAULT_DAILY_GOAL_SECONDS
        try:
            return max(60, int(row[0]))
        except ValueError:
            return DEFAULT_DAILY_GOAL_SECONDS

    def set_daily_goal_hours(self, hours):
        goal_seconds = int(float(hours) * 3600)
        if goal_seconds < 60:
            raise ValueError("Daily goal must be at least 1 minute.")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value)
                VALUES ('daily_goal_seconds', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(goal_seconds),),
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
        return [(day, int(total or 0)) for day, total in rows]

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

    def current_streak_days(self):
        goal = self.daily_goal_seconds()
        totals = dict(self.daily_totals(limit=365))
        streak = 0
        day = dt.date.today()
        while totals.get(day.isoformat(), 0) >= goal:
            streak += 1
            day -= dt.timedelta(days=1)
        return streak

    def export_sessions_csv(self, output_path):
        output_path = Path(output_path)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, started_at, ended_at, active_seconds, away_seconds, memo
                FROM sessions
                ORDER BY started_at DESC
                """
            ).fetchall()

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "started_at", "ended_at", "active_seconds", "away_seconds", "memo"])
            writer.writerows(rows)
        return output_path
