"""
רישום ריצות ונעילה לפי run_id (ה-76): רק חידוש אחד לכל ריצה בכל רגע - גם משני חלונות או משני תהליכים.
הטבלה יושבת בקובץ השמירה (checkpoints.db): הריצות שייכות לשמירות, והאיפוס מוחק את שתיהן יחד.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

STALE_AFTER = timedelta(minutes=15)  # נעילה ישנה מזה = התוכנית קרסה באמצע; מותר לתפוס מחדש (ערך מומצא)


class RunStatus(StrEnum):
    """מצב ריצה בטבלה."""

    WAITING = "WAITING"  # עצרה לאישור - מותר לחדש
    RESUMING = "RESUMING"  # חידוש בתהליך - נעולה
    DONE = "DONE"  # הסתיימה


class RunBusyError(Exception):
    """הריצה כבר בחידוש, הסתיימה, או לא קיימת - אי אפשר לתפוס אותה."""


class RunRegistry:
    """טבלת הריצות: מי מחכה, מי נעולה, מי הסתיימה."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path  # אותו קובץ של השומר
        db_path.parent.mkdir(parents=True, exist_ok=True)  # יוצר את התיקייה אם אין
        with closing(self._connect()) as conn, conn:  # conn בתוך with = שמירה (commit) בסוף, ביטול בשגיאה
            conn.execute(
                "CREATE TABLE IF NOT EXISTS run_registry ("
                " run_id TEXT PRIMARY KEY,"  # מזהה הריצה - פעם אחת בלבד
                " status TEXT NOT NULL,"  # WAITING / RESUMING / DONE
                " updated_at TEXT NOT NULL)"  # מתי המצב השתנה - לזיהוי נעילה נטושה
            )

    def mark_waiting(self, run_id: str) -> None:
        """הריצה עצרה לאישור - מותר לחדש אותה."""
        self._set(run_id, RunStatus.WAITING)

    def claim(self, run_id: str) -> None:
        """תופס את הריצה לחידוש - בפקודה אחת עם תנאי, כך שרק בקשה אחת מצליחה."""
        now = datetime.now()
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "UPDATE run_registry SET status = ?, updated_at = ?"
                " WHERE run_id = ? AND (status = ? OR (status = ? AND updated_at < ?))",
                (
                    RunStatus.RESUMING,  # המצב החדש: נעולה
                    now.isoformat(),  # מתי ננעלה
                    run_id,  # איזו ריצה
                    RunStatus.WAITING,  # מותר: מחכה לאישור
                    RunStatus.RESUMING,  # או: נעולה...
                    (now - STALE_AFTER).isoformat(),  # ...אבל מלפני יותר מ-15 דקות (נטושה)
                ),
            )
            claimed = cursor.rowcount == 1  # שורה אחת עודכנה = תפסנו
        if not claimed:  # אף שורה לא עודכנה - מישהו אחר תפס, או שהריצה לא מחכה
            raise RunBusyError(f"הריצה {run_id} כבר בטיפול, הסתיימה, או אינה קיימת")

    def release(self, run_id: str, status: RunStatus) -> None:
        """משחרר את הנעילה: חזרה ל"מחכה" (נכשל / עצר שוב) או "הסתיימה"."""
        self._set(run_id, status)

    def status(self, run_id: str) -> RunStatus | None:
        """מצב הריצה (None = לא רשומה). ישמש את מסך האישור (שלב 8) ואת הבדיקות."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT status FROM run_registry WHERE run_id = ?", (run_id,)).fetchone()
        return RunStatus(row[0]) if row else None

    def _set(self, run_id: str, status: RunStatus) -> None:
        """עזר: כותב מצב לריצה (חדשה או קיימת)."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO run_registry (run_id, status, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(run_id) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at",
                (run_id, status, datetime.now().isoformat()),  # מזהה, מצב, מתי
            )

    def _connect(self) -> sqlite3.Connection:
        """עזר: חיבור חדש לכל פעולה - בטוח לשימוש מכמה חלונות ותהליכים."""
        return sqlite3.connect(self._db_path, timeout=30)  # מחכה עד 30 שנ' אם הקובץ נעול לרגע
