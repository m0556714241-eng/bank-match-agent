"""
הגדרות הפרויקט במקום אחד: נתיבים, ערכים שאפשר לשנות, ומשתני הסביבה (.env).

כל קוד שצריך נתיב או הגדרה מקבל אותם מכאן - לא כותב אותם בעצמו.
המפתחות הסודיים נקראים מקובץ .env ולא נכתבים בקוד (החלטה מתחילת שלב 3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import truststore
from dotenv import load_dotenv

# תיקיית הפרויקט הראשית (שתי רמות מעל הקובץ הזה: src/config.py -> bank-match-agent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """כל ההגדרות. בלתי ניתן לשינוי אחרי שנטען."""

    # ---------- סביבה ----------
    app_env: str  # development = סביבת בדיקות, production = מערכת אמיתית

    # ---------- נתיבי קלט ----------
    bank_dir: Path  # קבצי הבנק היומיים
    customers_csv: Path
    invoices_csv: Path
    aliases_csv: Path  # הכינויים ההתחלתיים

    # ---------- נתיבים שהסוכן כותב אליהם ----------
    fx_cache_csv: Path  # מטמון השערים של בנק ישראל
    db_path: Path  # המסד: הנהלת החשבונות + הזיכרון

    # ---------- ערכים שאפשר לשנות ----------
    invoice_number_pattern: str  # תבנית מספר חשבונית בטקסט הבנק (החלטה ה-40)
    date_tolerance_days: int  # כמה ימים אחורה תאריך תנועה תקין ביחס ליום הקובץ
    fx_lookback_days: int  # כמה ימים אחורה מחפשים שער שפורסם

    # ---------- מודל השפה ומעקב (מקובץ .env) ----------
    anthropic_model: str
    anthropic_api_key: str | None  # None = עוד לא הוכנס מפתח (נדרש רק משלב 5)
    langsmith_project: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def use_system_certificates() -> None:
    """פייתון יבדוק תעודות אבטחה מול המאגר של מערכת ההפעלה (שמכיר את NetFree) - האבטחה נשארת מלאה."""
    # בלי זה: פייתון משתמש במאגר תעודות משלו, ונכשל מאחורי סינון כמו NetFree
    # (בנק ישראל, Anthropic, LangSmith). חל על כל התוכנית. החלטה ה-43.
    truststore.inject_into_ssl()


def load_settings(env_file: Path | None = None) -> Settings:
    """טוען את קובץ הסביבה ובונה את ההגדרות. נקרא פעם אחת בתחילת כל ריצה."""
    use_system_certificates()
    load_dotenv(env_file or PROJECT_ROOT / ".env")

    app_env = os.getenv("APP_ENV", "development").strip().lower()
    if app_env not in ("development", "production"):
        raise ValueError(f"APP_ENV חייב להיות development או production, התקבל: {app_env}")

    data = PROJECT_ROOT / "data"
    return Settings(
        app_env=app_env,
        bank_dir=data / "input" / "bank_transactions",
        customers_csv=data / "input" / "customers.csv",
        invoices_csv=data / "input" / "open_invoices.csv",
        aliases_csv=data / "memory" / "aliases.csv",
        fx_cache_csv=data / "input" / "fx_rates.csv",
        db_path=data / "output" / "ledger.db",
        invoice_number_pattern=r"INV-\d{5}",
        date_tolerance_days=4,
        fx_lookback_days=10,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,  # שדה ריק = אין מפתח
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "bank-match-agent"),
    )
