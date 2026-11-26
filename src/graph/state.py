"""
מצב הגרף (ה-71, ה-74): "הלוח המשותף" שכל הצמתים קוראים ממנו וכותבים אליו.

שני לוחות:
- RunState – הלוח הראשי של ריצה אחת (יום בנק אחד).
- TxState – הלוח הקטן של העברה אחת, בתוך התת-גרף.
ועוד המבנים שעוברים בין הגרף למסך האישור (שלב 8).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Annotated, TypedDict

from src.adapters.interfaces import Adjustment, Certainty, IntakeError, IntakeWarning
from src.domain.frozen import freeze_sequences
from src.domain.models import BankTransaction
from src.domain.money import Money
from src.engine.results import CustomerMatch, FilterResult, InvoiceMatch
from src.llm.customer_llm import LlmCustomerSuggestion
from src.llm.explain import ExplanationResult

# ---------- התוצאה של העברה אחת ----------


@dataclass(frozen=True)
class TxResult:
    """מה שהתת-גרף מוסיף ללוח הראשי בסוף הטיפול בהעברה אחת."""

    tx: BankTransaction  # ההעברה
    filter: FilterResult  # תוצאת הסינון
    customer: CustomerMatch | None = None  # תוצאת הזיהוי (None = סוננה לפני הזיהוי)
    llm: LlmCustomerSuggestion | None = None  # הצעת המודל (None = לא הגענו למודל)
    match: InvoiceMatch | None = None  # ההצעה או החריג (None = סוננה - אינה תקבול)


# ---------- מה שעובר למסך האישור ובחזרה (ה-74) ----------


class ApprovalAction(StrEnum):
    """ההחלטות האפשריות של מנהלת החשבונות על שורה."""

    APPROVE = "APPROVE"  # לאשר את ההצעה כמו שהיא
    REJECT = "REJECT"  # לדחות - לא נרשם כלום
    APPLY_TO_OLDEST = "APPLY_TO_OLDEST"  # חלקי בלי מספר חשבונית: להחיל על הוותיקה (ה-57)
    LEAVE_ON_ACCOUNT = "LEAVE_ON_ACCOUNT"  # להשאיר כזכות בחשבון הלקוח (ה-57; הרישום - שלב 7)


@dataclass(frozen=True)
class ApprovalLine:
    """שורת חשבונית אחת בתוך הצעה - מה בדיוק מאשרים."""

    invoice_id: str  # מספר החשבונית המוצעת
    open_before: Money  # היתרה הפתוחה לפני
    applied: Money  # הסכום שמוחל עליה
    open_after: Money  # היתרה אחרי


@dataclass(frozen=True)
class ApprovalRow:
    """שורה אחת במסך האישור."""

    tx_id: str  # מזהה ההעברה
    certainty: Certainty  # ודאית / סבירה / חריג
    customer_name: str | None  # שם הלקוח (None = לא זוהה)
    bank_amount: Money  # הסכום בבנק
    sentence: str  # משפט ההסבר (תבנית או מודל)
    lines: tuple[ApprovalLine, ...] = ()  # החשבוניות - שורה לכל חשבונית (ריק בחריג)
    adjustments: tuple[Adjustment, ...] = ()  # הפרשים: עמלה / ניכוי / הפרש שער
    allowed_actions: tuple[ApprovalAction, ...] = (ApprovalAction.APPROVE, ApprovalAction.REJECT)  # מה מותר לבחור

    def __post_init__(self) -> None:
        freeze_sequences(self)  # tuple נשאר tuple גם אחרי טעינה מקובץ השמירה (ה-71)


@dataclass(frozen=True)
class Decision:
    """ההחלטה של מנהלת החשבונות על שורה אחת - מה שהמסך מחזיר לגרף."""

    tx_id: str  # על איזו העברה
    action: ApprovalAction  # מה הוחלט
    approved_by: str  # מי החליט (נשמר ברישום)


# ---------- הלוח של העברה אחת (תת-גרף) ----------


class TxState(TypedDict, total=False):
    """הלוח הקטן: כל צומת בתת-גרף ממלא שדה אחד."""

    tx: BankTransaction  # ההעברה (נכנסת עם הפיצול)
    filter: FilterResult  # מ-filter_tx
    customer: CustomerMatch  # מ-identify_customer / verify_llm_customer
    llm: LlmCustomerSuggestion  # מ-llm_identify
    match: InvoiceMatch  # מ-find_invoice
    results: Annotated[list[TxResult], operator.add]  # היציאה: תוצאה אחת שעולה ללוח הראשי


class TxOutput(TypedDict):
    """מה תת-הגרף מחזיר לגרף הראשי - רק התוצאה (ה-74). כל שאר השדות נשארים פנימיים."""

    results: Annotated[list[TxResult], operator.add]  # אותו שדה ואותו כלל איחוד כמו ב-RunState - כדי שהשורות יצטרפו


# ---------- הלוח הראשי של ריצה ----------


class RunState(TypedDict, total=False):
    """הלוח הראשי של ריצה אחת."""

    run_id: str  # מזהה הריצה: run-<יום הבנק>-<תאריך ושעת ההפעלה> (ה-74)
    run_date: date  # איזה יום בנק מעבדים
    transactions: list[BankTransaction]  # ההעברות שעוד לא טופלו (מ-load_day)
    customer_names: dict[str, str]  # מזהה לקוח -> שם, לתצוגה ולהסברים (מ-load_day)
    intake_issues: list[IntakeError | IntakeWarning]  # שגיאות ואזהרות קליטה - לדוח (ה-39, ה-46)
    results: Annotated[list[TxResult], operator.add]  # כלל איחוד: כל העברה מוסיפה תוצאה (ה-71)
    checked_results: list[TxResult]  # התוצאות אחרי בדיקת ההתנגשות (מ-collect_results) - בלי כלל איחוד, מחליף
    explanations: ExplanationResult  # המשפטים והסיכום (מ-explain_and_summarize)
    approval_rows: list[ApprovalRow]  # מה שמוצג במסך האישור
    decisions: list[Decision]  # מה מנהלת החשבונות החליטה (מ-human_approval)
