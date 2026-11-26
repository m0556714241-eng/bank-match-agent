"""
הניתוב של הגרף (ה-49): "לאן ממשיכים" אחרי צומת שיש בו פיצול דרכים.

כל פונקציה כאן מקבלת את הזיכרון ומחזירה את שם הצומת הבא (או רשימת Send לפיצול).
אין כאן עבודה - רק החלטה על הדרך. החיבור לגרף - ב-build.py (add_conditional_edges).
"""

from __future__ import annotations

from typing import Literal  # סוג החזרה שמגדיר רשימה סגורה של ערכים מותרים

from langgraph.types import Send  # הוראה ל-LangGraph: "הפעל צומת X עם הקלט הזה"

from src.engine.results import CustomerOutcome, FilterOutcome  # ה-Enum-ים של תוצאת הזיהוי ותוצאת הסינון
from src.graph.state import RunState, TxState  # מבני הזיכרון: של כל הריצה, ושל העברה אחת


def route_after_filter(state: TxState) -> Literal["identify_customer", "finish_tx"]:
    """אחרי הסינון: תקבול -> לזיהוי לקוח; סוננה / לא נתמכת -> ישר לאריזה."""
    if state["filter"].outcome == FilterOutcome.PASS:  # הסינון כתב: "זה תקבול מלקוח"
        return "identify_customer"  # ממשיכים לזיהוי לקוח
    return "finish_tx"  # לא תקבול, או סוג שלא נתמך - finish_tx יחליט מה לכתוב


def route_after_identify(state: TxState) -> Literal["find_invoice", "llm_identify", "finish_tx"]:
    """אחרי הזיהוי בקוד: זוהה -> לחשבונית; לא בטוח -> למודל."""
    outcome = state["customer"].outcome  # מה צומת הזיהוי כתב: זוהה? צריך מודל?
    if outcome == CustomerOutcome.IDENTIFIED:  # זוהה בוודאות
        return "find_invoice"  # מחפשים חשבונית
    if outcome == CustomerOutcome.NEEDS_LLM:  # יש מועמדים אבל לא ודאי
        return "llm_identify"  # שואלים את המודל
    return "finish_tx"  # לא אמור לקרות כאן - ליתר ביטחון: חריג


def route_after_verify(state: TxState) -> Literal["find_invoice", "finish_tx"]:
    """אחרי אימות הצעת המודל: אומת -> לחשבונית; לא אומת -> לאריזה (חריג "לקוח לא זוהה")."""
    if state["customer"].outcome == CustomerOutcome.IDENTIFIED:  # האימות בקוד אישר את הצעת המודל
        return "find_invoice"  # מחפשים חשבונית
    return "finish_tx"  # לא אושר - אריזה כחריג


def fan_out(state: RunState) -> list[Send] | Literal["collect_results"]:
    """הפיצול: לכל העברה - תת-גרף משלו, במקביל (ה-49). אין העברות -> ישר לאיסוף."""
    transactions = state.get("transactions", [])  # ההעברות שעוד לא טופלו (מ-load_day)
    if not transactions:
        return "collect_results"  # יום בלי העברות חדשות - ישר לאיסוף (דוח של 0 העברות)
    return [
        Send("process_tx", {"tx": tx})  # תת-גרף "process_tx" מקבל רק את ההעברה שלו
        for tx in transactions  # הוראה אחת לכל העברה - עותק נפרד של תת-הגרף
    ]
