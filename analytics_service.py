import datetime as dt
import sqlite3
from pathlib import Path


class AnalyticsService:
    def __init__(self, db_path):
        self.db_path = Path(db_path)

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _day_range(self, days):
        today = dt.date.today()
        start_date = today - dt.timedelta(days=days - 1)
        end_date = today + dt.timedelta(days=1)
        return start_date, end_date

    def last_n_days_totals(self, days):
        start_date, end_date = self._day_range(days)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT substr(started_at, 1, 10) AS day, SUM(active_seconds) AS total
                FROM sessions
                WHERE started_at >= ? AND started_at < ?
                GROUP BY day
                ORDER BY day ASC
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchall()

        totals = {dt.date.fromisoformat(day): int(total or 0) for day, total in rows}
        result = []
        for i in range(days):
            current = start_date + dt.timedelta(days=i)
            result.append((current.isoformat(), totals.get(current, 0)))
        return result

    def session_count(self, days):
        start_date, end_date = self._day_range(days)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM sessions
                WHERE started_at >= ? AND started_at < ?
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchone()
        return int(row[0] or 0)

    def average_session_length(self, days):
        start_date, end_date = self._day_range(days)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT AVG(active_seconds)
                FROM sessions
                WHERE started_at >= ? AND started_at < ?
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchone()
        return int(row[0] or 0) if row and row[0] else 0

    def best_day(self, days):
        totals = self.last_n_days_totals(days)
        if not totals:
            return None, 0
        best_day, best_seconds = max(totals, key=lambda item: item[1])
        return best_day, best_seconds

    def summary_for_days(self, days, daily_goal_seconds):
        totals = self.last_n_days_totals(days)
        total_time = sum(value for _, value in totals)
        average_daily = int(total_time / days) if days else 0
        session_count = self.session_count(days)
        average_session = self.average_session_length(days)
        best_date, best_seconds = self.best_day(days)
        goal_ratio = 0
        if daily_goal_seconds and days:
            goal_ratio = min(100.0, round(total_time / (daily_goal_seconds * days) * 100, 1))
        return {
            "totals": totals,
            "total_time": total_time,
            "average_daily": average_daily,
            "session_count": session_count,
            "average_session": average_session,
            "best_date": best_date,
            "best_seconds": best_seconds,
            "goal_ratio": goal_ratio,
        }
