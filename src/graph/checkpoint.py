"""
שמירת מצב הגרף לקובץ (ה-71).

השומר כותב את המצב לקובץ SQLite אחרי כל צעד, כדי שאפשר יהיה לעצור לאישור ולהמשיך – גם אחרי שעות.
"האורז" שלו בונה מחדש רק מחלקות שברשימת ההיתר – הגנה מפני קובץ שמירה שמישהו שינה.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# רשימת ההיתר: (מודול, מחלקה) – רק אלה מותר לבנות מחדש בטעינה מהקובץ
ALLOWED_TYPES: list[tuple[str, str]] = [
    # כסף והעברות
    ("src.domain.money", "Currency"),
    ("src.domain.money", "Money"),
    ("src.domain.models", "BankTransaction"),
    # מהממשקים: ודאות, החלות, הפרשים, בעיות קליטה, תוצאת רישום
    ("src.adapters.interfaces", "Certainty"),
    ("src.adapters.interfaces", "Application"),
    ("src.adapters.interfaces", "Adjustment"),
    ("src.adapters.interfaces", "AdjustmentType"),
    ("src.adapters.interfaces", "IntakeError"),
    ("src.adapters.interfaces", "IntakeErrorCode"),
    ("src.adapters.interfaces", "IntakeWarning"),
    ("src.adapters.interfaces", "IntakeWarningCode"),
    ("src.adapters.interfaces", "PostingResult"),
    # תוצאות המנוע
    ("src.engine.results", "FilterResult"),
    ("src.engine.results", "FilterOutcome"),
    ("src.engine.results", "FilterReason"),
    ("src.engine.results", "Candidate"),
    ("src.engine.results", "CustomerMatch"),
    ("src.engine.results", "CustomerOutcome"),
    ("src.engine.results", "IdentifiedBy"),
    ("src.engine.results", "InvoiceMatch"),
    ("src.engine.results", "MatchKind"),
    ("src.engine.results", "ExceptionReason"),
    # תוצאות המודל
    ("src.llm.customer_llm", "LlmCustomerSuggestion"),
    ("src.llm.explain", "RunTotals"),
    ("src.llm.explain", "ExplanationResult"),
    # מצב הגרף ומסך האישור
    ("src.graph.state", "TxResult"),
    ("src.graph.state", "ApprovalAction"),
    ("src.graph.state", "ApprovalLine"),
    ("src.graph.state", "ApprovalRow"),
    ("src.graph.state", "Decision"),
    ("src.graph.state", "PostingStatus"),
    ("src.graph.state", "PostingOutcome"),
]


def make_serializer() -> JsonPlusSerializer:
    """"האורז" של LangGraph, עם רשימת ההיתר שלנו."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)


def open_checkpointer(db_path: Path) -> SqliteSaver:
    """פותח שומר לקובץ. אותו קובץ + אותו מזהה ריצה = ממשיכים מאיפה שעצרנו."""
    db_path.parent.mkdir(parents=True, exist_ok=True)  # יוצר את התיקייה אם אין
    conn = sqlite3.connect(db_path, check_same_thread=False)  # הגרף יכול לכתוב מכמה תהליכונים
    return SqliteSaver(conn, serde=make_serializer())
