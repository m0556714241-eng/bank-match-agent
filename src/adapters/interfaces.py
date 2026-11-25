"""
הממשקים - מה כל מתאם חייב לספק (החלטה ה-32).

המנוע והגרף מכירים רק את הממשקים האלה, לא את המימוש.
היום: קבצים ו-SQLite. מחר: Nilus ו-NetSuite - בלי לשנות את המנוע (החלטה ה-11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Protocol

from src.domain.models import Alias, BankTransaction, Customer, FxRate, Invoice, OpenInvoice
from src.domain.money import Currency, Money

# ---------- סוגים משותפים לממשקים ----------


class IntakeErrorCode(StrEnum):
    """קודי שגיאות קליטה (החלטה ה-41). שגיאת קליטה אינה חריג - היא שורה פגומה בקובץ."""

    INVALID_AMOUNT_PRECISION = "INVALID_AMOUNT_PRECISION"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    UNKNOWN_CURRENCY = "UNKNOWN_CURRENCY"
    INVALID_DATE = "INVALID_DATE"
    UNKNOWN_REFERENCE = "UNKNOWN_REFERENCE"
    MISSING_FIELD = "MISSING_FIELD"
    DUPLICATE_TX_ID = "DUPLICATE_TX_ID"


@dataclass(frozen=True)
class IntakeError:
    """שורה אחת שלא נקלטה - לדוח הריצה."""

    source: str  # מאיזה קובץ
    row_number: int  # איזו שורה בקובץ (כדי שאדם ימצא אותה)
    code: IntakeErrorCode
    message: str  # הסבר קצר בעברית
    raw: dict[str, str]  # השורה המקורית, כפי שהופיעה


class IntakeWarningCode(StrEnum):
    """סוגי אזהרות קליטה - השורה נקלטה, אבל כדאי שאדם יידע."""

    AMOUNT_ROUNDED = "AMOUNT_ROUNDED"  # סכום עוגל (רעש של Excel)
    DATE_OUT_OF_RANGE = "DATE_OUT_OF_RANGE"  # תאריך ישן מדי או עתידי ביחס ליום הקובץ
    FILE_MISSING = "FILE_MISSING"  # לא הגיע קובץ ליום הזה


@dataclass(frozen=True)
class IntakeWarning:
    """אזהרת קליטה אחת - אובייקט עם סוג, כדי שמסך האישור יוכל לקבץ ולהציג."""

    source: str  # מאיזה קובץ
    row_number: int | None  # איזו שורה (None כשהאזהרה על הקובץ כולו)
    code: IntakeWarningCode
    message: str  # הסבר קצר בעברית


@dataclass(frozen=True)
class BankLoadResult:
    """מה שמתאם הבנק מחזיר: התנועות שנקלטו, השורות שלא נקלטו, ואזהרות."""

    transactions: list[BankTransaction]
    intake_errors: list[IntakeError] = field(default_factory=list)
    warnings: list[IntakeWarning] = field(default_factory=list)


class Certainty(StrEnum):
    """שלוש רמות הוודאות (החלטה ה-10)."""

    CERTAIN = "CERTAIN"
    PROBABLE = "PROBABLE"
    EXCEPTION = "EXCEPTION"


class AdjustmentType(StrEnum):
    """סוגי ההפרשים - רשימה סגורה, זהה להגבלה במסד (החלטה ה-37)."""

    BANK_FEE = "BANK_FEE"
    WITHHOLDING_TAX = "WITHHOLDING_TAX"
    FX_DIFFERENCE = "FX_DIFFERENCE"
    ROUNDING = "ROUNDING"


@dataclass(frozen=True)
class Application:
    """החלה אחת: כמה מהתקבול הולך לאיזו חשבונית."""

    invoice_id: str
    amount: Money


@dataclass(frozen=True)
class Adjustment:
    """הפרש אחד, עם הסוג שלו."""

    type: AdjustmentType
    amount: Money


@dataclass(frozen=True)
class ReceiptPosting:
    """בקשת רישום: כל מה שנרשם יחד, כיחידה אחת (החלטה ה-33)."""

    tx_id: str
    customer_id: str
    amount: Money  # סכום התקבול
    certainty: Certainty
    approved_by: str
    run_id: str
    applications: list[Application]
    adjustments: list[Adjustment] = field(default_factory=list)

    def __post_init__(self) -> None:
        # בדיקות בסיס - עוד לפני שמגיעים למסד
        if not self.applications:
            raise ValueError(f"{self.tx_id}: רישום בלי החלה על חשבונית")
        total = Money.zero(self.amount.currency)
        for app in self.applications:
            total = total + app.amount  # זורק שגיאה אם המטבע שונה
        if total != self.amount:
            raise ValueError(f"{self.tx_id}: סכום ההחלות {total} שונה מסכום התקבול {self.amount}")


@dataclass(frozen=True)
class PostingResult:
    """תוצאת רישום מוצלח: מספר התקבול והיתרות אחרי הרישום (לבדיקת הבקרה)."""

    receipt_id: str
    balances_after: dict[str, Money]  # invoice_id -> יתרה פתוחה אחרי הרישום


# ---------- שגיאות שהממשקים מגדירים ----------


class DuplicatePostingError(Exception):
    """ניסיון לרשום תקבול שני מאותה תנועת בנק (החלטה ה-12)."""


class PostingControlError(Exception):
    """בדיקת הבקרה נכשלה: היתרה לא ירדה בדיוק כמתוכנן. שום דבר לא נרשם."""


class FxRateUnavailable(Exception):
    """אין שער ליום הזה - לא במטמון ולא אצל בנק ישראל. התנועה תהפוך לחריג."""


# ---------- שלושת הממשקים ----------


class BankSource(Protocol):
    """מקור תנועות הבנק. היום: קבצים יומיים. בעתיד: הייצוא היומי של Nilus."""

    def load_day(self, day: date) -> BankLoadResult:
        """כל התנועות של יום אחד, אחרי בדיקת תקינות."""
        ...


class LedgerStore(Protocol):
    """מערכת הנהלת החשבונות + הזיכרון של הסוכן. היום: SQLite. בעתיד: NetSuite."""

    def customers(self) -> list[Customer]:
        """כל הלקוחות."""
        ...

    def open_invoices(self, as_of: date, customer_id: str | None = None) -> list[OpenInvoice]:
        """חשבוניות עם יתרה פתוחה, שהופקו עד התאריך (החלטה ה-20). אפשר לסנן לפי לקוח."""
        ...

    def paid_invoices(self, as_of: date, customer_id: str | None = None) -> list[Invoice]:
        """חשבוניות ששולמו במלואן, שהופקו עד התאריך - לזיהוי תשלום כפול (ה-57)."""
        ...

    def aliases(self) -> list[Alias]:
        """כל הכינויים הידועים."""
        ...

    def confirm_alias(self, bank_name: str, customer_id: str) -> Alias:
        """אישור של כינוי: מוסיף אותו, או מעלה את מונה האישורים ב-1."""
        ...

    def processed_tx_ids(self, tx_ids: list[str]) -> set[str]:
        """מתוך רשימת תנועות - אילו כבר טופלו (הגנה מכפילות)."""
        ...

    def mark_processed(self, tx_id: str, run_id: str, result: str) -> None:
        """סימון תנועה שטופלה בלי רישום תקבול (למשל: סוננה, נדחתה)."""
        ...

    def post_receipt(self, posting: ReceiptPosting) -> PostingResult:
        """רישום תקבול + החלות + הפרשים + סימון כמטופלת - הכול או כלום, כולל בדיקת בקרה."""
        ...


class FxRateProvider(Protocol):
    """שערי מטבע. היום: בנק ישראל + מטמון בקובץ. בעתיד: טבלת השערים של NetSuite."""

    def rate(self, currency: Currency, on: date) -> FxRate:
        """השער היציג ליום. יום בלי פרסום - השער של יום העסקים הקודם (החלטה ה-22)."""
        ...
