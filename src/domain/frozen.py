"""
עזר לאובייקטים שאסור לשנות (ה-71).

כש-LangGraph שומר את מצב הגרף לקובץ וטוען אותו בחזרה, tuple חוזר כ-list.
list אפשר לשנות, ו-tuple לא – לכן מחזירים כל רשימה ל-tuple מיד כשהאובייקט נוצר.
"""

from __future__ import annotations

from dataclasses import fields


def freeze_sequences(obj: object) -> None:
    """הופכת כל שדה מסוג list ל-tuple, באובייקט dataclass שאי אפשר לשנות (frozen)."""
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, list):
            object.__setattr__(obj, f.name, tuple(value))  # frozen חוסם השמה רגילה - זו הדרך המקובלת
