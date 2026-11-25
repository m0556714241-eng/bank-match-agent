"""
יצירת מודל השפה (שלב 5, ה-62, ה-66).
מקום אחד שבו נקבעות ההגדרות של כל קריאה למודל: איזה מודל, תקרת אורך, זמן המתנה.
שני הצמתים של המודל (זיהוי לקוח, הסברים) מקבלים את המודל מכאן.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import Settings

if TYPE_CHECKING:  # רק לבדיקת טיפוסים - הספרייה נטענת בפועל רק בתוך הפונקציה
    from langchain_anthropic import ChatAnthropic

REQUEST_TIMEOUT_SECONDS = 60  # זמן המתנה לקריאה אחת (ה-66)


class MissingApiKeyError(RuntimeError):
    """אין מפתח Anthropic בקובץ .env - אי אפשר לקרוא למודל אמיתי."""


def make_chat_model(settings: Settings, max_tokens: int) -> ChatAnthropic:
    """יוצר מודל Claude עם ההגדרות של הפרויקט. max_tokens שונה לכל צומת (300 / 4,000)."""
    if not settings.anthropic_api_key:
        raise MissingApiKeyError("חסר ANTHROPIC_API_KEY בקובץ .env")

    from langchain_anthropic import ChatAnthropic  # טעינה רק כשבאמת צריך מודל אמיתי

    return ChatAnthropic(
        model=settings.anthropic_model,  # מ-.env, לא בקוד (ה-62)
        api_key=settings.anthropic_api_key,  # מ-.env, לא נשמר במצב הגרף
        max_tokens=max_tokens,  # תקרת אורך התשובה (ה-64)
        temperature=0,  # אותה תנועה -> אותה תשובה (ה-66)
        timeout=REQUEST_TIMEOUT_SECONDS,  # הבקשה עצמה נקטעת אחרי 60 שניות
        max_retries=0,  # ניסיון חוזר רק במקום אחד - ב-RetryPolicy של הצומת (ה-66)
    )
