"""
מה כל צומת של המנוע מחזיר. כל תוצאה היא אובייקט קטן ובלתי ניתן לשינוי,
שהגרף (שלב 6) קורא כדי להחליט לאן להמשיך (ענף מותנה).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from src.adapters.interfaces import Adjustment, Application, Certainty
from src.domain.frozen import freeze_sequences
from src.domain.money import Money

# ---------- קודי סיבה ----------


class FilterReason(StrEnum):
    """למה תנועה לא נכנסה להתאמה (סעיף 4)."""

    OUTGOING = "OUTGOING"  # תנועה יוצאת (סכום שלילי)
    INTERNAL_TRANSFER = "INTERNAL_TRANSFER"  # העברה בין חשבונות החברה
    INTEREST = "INTEREST"  # ריבית
    TAX_REFUND = "TAX_REFUND"  # החזר מס


class ExceptionReason(StrEnum):
    """למה תנועה היא חריג (סעיף 9), ועוד קוד אחד לשער חסר (ה-34)."""

    CUSTOMER_NOT_IDENTIFIED = "CUSTOMER_NOT_IDENTIFIED"  # לא הקוד ולא המודל זיהו לקוח
    LLM_SUGGESTION_NOT_VERIFIED = "LLM_SUGGESTION_NOT_VERIFIED"  # המודל הציע לקוח, ואין לו חשבונית מתאימה
    DIFF_NOT_EXPLAINED = "DIFF_NOT_EXPLAINED"  # לקוח ידוע, אבל ההפרש לא מוסבר
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"  # שני לקוחות מתאימים באותה מידה
    OVERPAYMENT = "OVERPAYMENT"  # יותר מכל החשבוניות הפתוחות / תשלום כפול
    PARTIAL_WITHOUT_REFERENCE = "PARTIAL_WITHOUT_REFERENCE"  # סכום חלקי בלי מספר חשבונית
    FX_DIFF_ABOVE_TOLERANCE = "FX_DIFF_ABOVE_TOLERANCE"  # הפרש שער מעל 2%
    UNSUPPORTED_TX_TYPE = "UNSUPPORTED_TX_TYPE"  # סוג תנועה שלא נתמך (סליקת אשראי)
    FX_RATE_UNAVAILABLE = "FX_RATE_UNAVAILABLE"  # אין שער לתאריך (לא מנחשים)
    INVOICE_CONFLICT = "INVOICE_CONFLICT"  # אותה חשבונית כבר הוצעה להעברה אחרת באותה ריצה (ה-71)


# ---------- צומת 1: סינון ----------


class FilterOutcome(StrEnum):
    """תוצאת הסינון - כל ערך הוא ענף אחר בגרף."""

    PASS = "PASS"  # תקבול -> ממשיכים לזיהוי לקוח
    FILTERED = "FILTERED"  # לא תקבול -> מסמנים ומדלגים
    UNSUPPORTED = "UNSUPPORTED"  # סוג שלא נתמך -> חריג לאדם


@dataclass(frozen=True)
class FilterResult:
    """מה צומת הסינון מחזיר."""

    outcome: FilterOutcome  # לאן ממשיכים
    filter_reason: FilterReason | None = None  # למה סוננה (רק כשסוננה)


# ---------- צומת 2: זיהוי לקוח ----------


class IdentifiedBy(StrEnum):
    """איך הלקוח זוהה (סעיף 5) - משפיע על רמת הוודאות."""

    INVOICE_REF = "INVOICE_REF"  # מספר חשבונית בטקסט
    EXACT_NAME = "EXACT_NAME"  # שם זהה לשם הרשמי (אחרי נרמול)
    ALIAS = "ALIAS"  # כינוי מוכר
    FUZZY = "FUZZY"  # שם דומה (ציון 90+)
    LLM = "LLM"  # מודל השפה (שלב 5), אחרי אימות בקוד


class CustomerOutcome(StrEnum):
    """תוצאת הזיהוי - כל ערך הוא ענף אחר בגרף."""

    IDENTIFIED = "IDENTIFIED"  # זוהה -> למציאת חשבונית
    NEEDS_LLM = "NEEDS_LLM"  # הקוד לא בטוח -> לצומת של המודל
    NOT_VERIFIED = "NOT_VERIFIED"  # המודל הציע לקוח שלא קיים -> חריג


@dataclass(frozen=True)
class Candidate:
    """לקוח מועמד - מה שהמודל יקבל כשהקוד לא בטוח."""

    customer_id: str  # מזהה הלקוח
    score: float  # ציון הדמיון לטקסט הבנק (0–100)


@dataclass(frozen=True)
class CustomerMatch:
    """מה צומת הזיהוי מחזיר."""

    outcome: CustomerOutcome  # לאן ממשיכים
    customer_ids: tuple[str, ...] = ()  # מי זוהה. יותר מאחד = כינוי משותף (פייפלו), הסכום יכריע
    identified_by: IdentifiedBy | None = None  # איך זוהה
    strong: bool = False  # זיהוי ודאי? (מספר חשבונית / שם זהה / כינוי שאושר 3 פעמים)
    candidates: tuple[Candidate, ...] = ()  # המועמדים למודל (רק כשצריך מודל)
    invoice_refs: tuple[str, ...] = ()  # מספרי חשבונית שנמצאו בטקסט הבנק

    def __post_init__(self) -> None:
        freeze_sequences(self)  # tuple נשאר tuple גם אחרי טעינה מקובץ השמירה (ה-71)


# ---------- צומת 3: מציאת חשבונית ----------


class MatchKind(StrEnum):
    """איזה כלל מסעיף 6 מצא את ההתאמה."""

    EXACT_REF = "EXACT_REF"  # מספר חשבונית בטקסט + סכום זהה
    EXACT = "EXACT"  # חשבונית אחת בסכום זהה
    COMBINATION = "COMBINATION"  # כמה חשבוניות שביחד שוות לסכום
    BANK_FEE = "BANK_FEE"  # חסר בגלל עמלת בנק
    WITHHOLDING = "WITHHOLDING"  # חסר בגלל ניכוי במקור
    FX = "FX"  # חשבונית במט"ח ששולמה בשקלים
    PARTIAL_REF = "PARTIAL_REF"  # תשלום חלקי עם מספר חשבונית


@dataclass(frozen=True)
class InvoiceMatch:
    """מה צומת מציאת החשבונית מחזיר: הצעה לרישום, או חריג עם הסיבה."""

    certainty: Certainty  # ודאית / סבירה / חריג
    customer_id: str | None = None  # מי שילם
    kind: MatchKind | None = None  # איזה כלל מצא (ריק בחריג)
    receipt_amount: Money | None = None  # סכום התקבול, במטבע החשבונית
    applications: tuple[Application, ...] = ()  # על אילו חשבוניות, וכמה על כל אחת
    adjustments: tuple[Adjustment, ...] = ()  # הפרשים: סוג וסכום
    exception_reason: ExceptionReason | None = None  # למה חריג (ריק בהצעה)
    notes: tuple[str, ...] = ()  # הערות לאדם, למשל "היו עוד אפשרויות - הוצעה הוותיקה"
    evidence: dict[str, str] = field(default_factory=dict)  # העובדות - ממנו המודל ינסח את משפט ההסבר

    def __post_init__(self) -> None:
        freeze_sequences(self)  # tuple נשאר tuple גם אחרי טעינה מקובץ השמירה (ה-71)

    @property
    def is_exception(self) -> bool:
        """קיצור: האם זה חריג."""
        return self.certainty == Certainty.EXCEPTION
