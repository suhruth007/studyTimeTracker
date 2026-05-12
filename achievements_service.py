import datetime as dt
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class AchievementService:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            # Achievements table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS achievements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    icon TEXT NOT NULL,
                    category TEXT NOT NULL,
                    unlocked_at TEXT,
                    progress INTEGER DEFAULT 0,
                    target INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Achievement unlocks table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS achievement_unlocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    achievement_id INTEGER NOT NULL,
                    unlocked_at TEXT NOT NULL,
                    session_id INTEGER,
                    FOREIGN KEY (achievement_id) REFERENCES achievements (id),
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
                """
            )

            # Initialize default achievements
            self._init_default_achievements()

    def _init_default_achievements(self):
        achievements = [
            # Study Streaks
            ("First Steps", "Complete your first study session", "🎯", "streak", 1),
            ("Getting Started", "Study for 7 consecutive days", "🔥", "streak", 7),
            ("Dedicated Learner", "Study for 30 consecutive days", "💪", "streak", 30),
            ("Study Master", "Study for 100 consecutive days", "👑", "streak", 100),
            ("Unstoppable", "Study for 365 consecutive days", "🚀", "streak", 365),

            # Productivity Milestones
            ("Hour Warrior", "Study for 1 hour in a single session", "⏰", "productivity", 3600),
            ("Marathon Runner", "Study for 4 hours in a single session", "🏃", "productivity", 14400),
            ("All-Nighter", "Study for 8 hours in a single session", "🌙", "productivity", 28800),
            ("Century Club", "Study for 100 hours total", "💯", "productivity", 360000),
            ("Thousand Hour Club", "Study for 1000 hours total", "🏆", "productivity", 3600000),

            # Daily Goals
            ("Goal Crusher", "Meet your daily goal 10 times", "🎯", "goals", 10),
            ("Goal Master", "Meet your daily goal 50 times", "🎯", "goals", 50),
            ("Goal Legend", "Meet your daily goal 100 times", "🎯", "goals", 100),

            # Special Achievements
            ("Early Bird", "Study before 6 AM", "🌅", "special", 1),
            ("Night Owl", "Study after 11 PM", "🦉", "special", 1),
            ("Weekend Warrior", "Study every day of a weekend", "📅", "special", 1),
            ("Consistency King", "Study at the same time for 7 days", "⚡", "special", 7),
        ]

        with self._connect() as conn:
            for name, desc, icon, category, target in achievements:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO achievements (name, description, icon, category, target)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, desc, icon, category, target)
                )

    def get_all_achievements(self) -> List[Dict]:
        """Get all achievements with unlock status."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.id, a.name, a.description, a.icon, a.category, a.target, a.progress,
                       au.unlocked_at, au.session_id
                FROM achievements a
                LEFT JOIN achievement_unlocks au ON a.id = au.achievement_id
                ORDER BY a.category, a.target
                """
            ).fetchall()

        achievements = []
        for row in rows:
            achievements.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'icon': row[3],
                'category': row[4],
                'target': row[5],
                'progress': row[6],
                'unlocked_at': row[7],
                'session_id': row[8],
                'is_unlocked': row[7] is not None
            })
        return achievements

    def get_unlocked_achievements(self) -> List[Dict]:
        """Get only unlocked achievements."""
        return [a for a in self.get_all_achievements() if a['is_unlocked']]

    def get_recent_unlocks(self, limit: int = 5) -> List[Dict]:
        """Get recently unlocked achievements."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT a.name, a.description, a.icon, au.unlocked_at
                FROM achievements a
                JOIN achievement_unlocks au ON a.id = au.achievement_id
                ORDER BY au.unlocked_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return [{
            'name': row[0],
            'description': row[1],
            'icon': row[2],
            'unlocked_at': row[3]
        } for row in rows]

    def check_and_update_achievements(self, stats_service, session_data: Optional[Dict] = None) -> List[Dict]:
        """Check for new achievements and update progress. Returns newly unlocked achievements."""
        newly_unlocked = []

        # Get current stats
        current_streak = stats_service.current_streak_days()
        total_seconds = sum(total for _, total in stats_service.daily_totals(limit=1000))
        today_total = stats_service.today_total()
        goal_days = self._count_goal_days_met(stats_service)

        # Check each achievement
        with self._connect() as conn:
            achievements = conn.execute(
                "SELECT id, name, category, target FROM achievements WHERE unlocked_at IS NULL"
            ).fetchall()

            for ach_id, name, category, target in achievements:
                progress = 0
                should_unlock = False

                if category == "streak":
                    progress = current_streak
                    should_unlock = current_streak >= target
                elif category == "productivity":
                    if "total" in name.lower():
                        progress = total_seconds
                        should_unlock = total_seconds >= target
                    else:
                        # Single session achievements
                        if session_data:
                            progress = session_data.get('active_seconds', 0)
                            should_unlock = progress >= target
                elif category == "goals":
                    progress = goal_days
                    should_unlock = goal_days >= target
                elif category == "special":
                    # Special achievements require specific conditions
                    should_unlock = self._check_special_achievement(name, stats_service, session_data)

                # Update progress
                conn.execute(
                    "UPDATE achievements SET progress = ? WHERE id = ?",
                    (progress, ach_id)
                )

                # Unlock if criteria met
                if should_unlock:
                    unlock_time = dt.datetime.now().isoformat()
                    session_id = session_data.get('id') if session_data else None

                    conn.execute(
                        """
                        INSERT INTO achievement_unlocks (achievement_id, unlocked_at, session_id)
                        VALUES (?, ?, ?)
                        """,
                        (ach_id, unlock_time, session_id)
                    )

                    conn.execute(
                        "UPDATE achievements SET unlocked_at = ? WHERE id = ?",
                        (unlock_time, ach_id)
                    )

                    newly_unlocked.append({
                        'name': name,
                        'icon': self._get_achievement_icon(name),
                        'description': self._get_achievement_description(name)
                    })

        return newly_unlocked

    def _count_goal_days_met(self, stats_service) -> int:
        """Count how many days the user has met their daily goal."""
        goal = stats_service.daily_goal_seconds()
        totals = dict(stats_service.daily_totals(limit=365))
        return sum(1 for total in totals.values() if total >= goal)

    def _check_special_achievement(self, name: str, stats_service, session_data: Optional[Dict]) -> bool:
        """Check special achievements that require specific conditions."""
        if name == "Early Bird":
            if session_data:
                start_time = dt.datetime.fromisoformat(session_data['started_at'])
                return start_time.hour < 6
        elif name == "Night Owl":
            if session_data:
                start_time = dt.datetime.fromisoformat(session_data['started_at'])
                return start_time.hour >= 23
        elif name == "Weekend Warrior":
            # Check if studied both Saturday and Sunday in any weekend
            totals = dict(stats_service.daily_totals(limit=365))
            for i in range(len(totals) - 1):
                days = list(totals.keys())[i:i+2]
                if len(days) == 2:
                    day1 = dt.datetime.fromisoformat(days[0]).date()
                    day2 = dt.datetime.fromisoformat(days[1]).date()
                    if day1.weekday() == 5 and day2.weekday() == 6:  # Sat-Sun
                        if totals[days[0]] > 0 and totals[days[1]] > 0:
                            return True
        elif name == "Consistency King":
            # Check for 7 consecutive days at similar times
            return self._check_time_consistency(stats_service, 7)

        return False

    def _check_time_consistency(self, stats_service, days_required: int) -> bool:
        """Check if user studied at consistent times for N consecutive days."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT substr(started_at, 1, 10) as day, strftime('%H', started_at) as hour
                FROM sessions
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (days_required * 5,)  # Allow some flexibility
            ).fetchall()

        if len(rows) < days_required:
            return False

        # Group by day and find most common hour per day
        day_hours = {}
        for day, hour in rows:
            if day not in day_hours:
                day_hours[day] = []
            day_hours[day].append(int(hour))

        # Get most common hour for each day
        consistent_days = 0
        prev_hour = None

        for day in sorted(day_hours.keys(), reverse=True):
            hours = day_hours[day]
            if hours:
                most_common = max(set(hours), key=hours.count)
                if prev_hour is None or abs(most_common - prev_hour) <= 1:  # Within 1 hour
                    consistent_days += 1
                    prev_hour = most_common
                else:
                    break

            if consistent_days >= days_required:
                return True

        return False

    def _get_achievement_icon(self, name: str) -> str:
        """Get achievement icon by name."""
        icons = {
            "First Steps": "🎯",
            "Getting Started": "🔥",
            "Dedicated Learner": "💪",
            "Study Master": "👑",
            "Unstoppable": "🚀",
            "Hour Warrior": "⏰",
            "Marathon Runner": "🏃",
            "All-Nighter": "🌙",
            "Century Club": "💯",
            "Thousand Hour Club": "🏆",
            "Goal Crusher": "🎯",
            "Goal Master": "🎯",
            "Goal Legend": "🎯",
            "Early Bird": "🌅",
            "Night Owl": "🦉",
            "Weekend Warrior": "📅",
            "Consistency King": "⚡"
        }
        return icons.get(name, "🏅")

    def _get_achievement_description(self, name: str) -> str:
        """Get achievement description by name."""
        descriptions = {
            "First Steps": "Complete your first study session",
            "Getting Started": "Study for 7 consecutive days",
            "Dedicated Learner": "Study for 30 consecutive days",
            "Study Master": "Study for 100 consecutive days",
            "Unstoppable": "Study for 365 consecutive days",
            "Hour Warrior": "Study for 1 hour in a single session",
            "Marathon Runner": "Study for 4 hours in a single session",
            "All-Nighter": "Study for 8 hours in a single session",
            "Century Club": "Study for 100 hours total",
            "Thousand Hour Club": "Study for 1000 hours total",
            "Goal Crusher": "Meet your daily goal 10 times",
            "Goal Master": "Meet your daily goal 50 times",
            "Goal Legend": "Meet your daily goal 100 times",
            "Early Bird": "Study before 6 AM",
            "Night Owl": "Study after 11 PM",
            "Weekend Warrior": "Study every day of a weekend",
            "Consistency King": "Study at the same time for 7 days"
        }
        return descriptions.get(name, "")

    def get_achievement_stats(self) -> Dict:
        """Get achievement statistics."""
        all_achievements = self.get_all_achievements()
        unlocked = [a for a in all_achievements if a['is_unlocked']]

        return {
            'total_achievements': len(all_achievements),
            'unlocked_count': len(unlocked),
            'completion_percentage': round(len(unlocked) / len(all_achievements) * 100, 1) if all_achievements else 0,
            'recent_unlocks': self.get_recent_unlocks(3)
        }
<parameter name="filePath">d:\MyCode\Learning\Quant\achievements_service.py
