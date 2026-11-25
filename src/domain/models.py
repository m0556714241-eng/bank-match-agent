"""
מבני הנתונים של הפרויקט - "השפה המשותפת" של המתאמים, המנוע ומודל השפה.

כל רשומה נבדקת בכניסה (החלטה ה-32): שורה פגומה בקובץ נעצרת כאן
עם הודעה ברורה, ולא מגיעה למנוע.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.domain.money import Currency, Money

# ---------- מזהים ----------
# מזהה = מחרוזת לא ריקה, עד 64 תווים, בלי תבנית קבועה (החלטה ה-40).
# כל מערכת מגדירה פורמט משלה, ולכן הבדיקה האמיתית היא שהמזהה קיים - בטעינה למסד.
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=64)]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class _Record(BaseModel):
    """בסיס לכל הרשומות: בלתי ניתנות לשינוי, ושדה לא מוכר = שגיאה."""

    model_config = ConfigDict(
        frozen=True,  # רשומה שנקראה לא משתנה בדרך
        extra="forbid",  # עמודה לא צפויה בקובץ = שגיאה, לא התעלמות שקטה
        str_strip_whitespace=True,  # רווחים מיותרים בקצוות נמחקים
        validate_by_name=True,  # אפשר ליצור לפי שם השדה בקוד...
        validate_by_alias=True,  # ...או לפי שם העמודה בקובץ (למשל "date")
    )


class Customer(_Record):
    """לקוח. מקבילה ב-NetSuite: customer."""

    customer_id: Identifier
    name: NonEmptyText
    currency: Currency
    withholding_rate: Annotated[Decimal, Field(ge=0, lt=1)]  # 0.03 = 3%


class Invoice(_Record):
    """חשבונית - העובדות שלא משתנות. מקבילה ב-NetSuite: invoice."""

    invoice_id: Identifier
    customer_id: Identifier
    issue_date: date = Field(alias="date")
    amount: Annotated[Decimal, Field(gt=0)]
    currency: Currency

    @model_validator(mode="after")
    def _amount_fits_currency(self) -> Invoice:
        # בדיקה שהסכום לא כולל יותר ספרות ממה שהמטבע מאפשר
        Money.of(self.amount, self.currency)
        return self

    @property
    def money(self) -> Money:
        return Money.of(self.amount, self.currency)


class OpenInvoice(Invoice):
    """חשבונית + היתרה הפתוחה שלה כרגע (כפי שהמסד חישב, או יתרת פתיחה מהקובץ)."""

    open_balance: Annotated[Decimal, Field(ge=0)]

    @model_validator(mode="after")
    def _open_not_above_amount(self) -> OpenInvoice:
        Money.of(self.open_balance, self.currency)
        if self.open_balance > self.amount:
            raise ValueError(
                f"{self.invoice_id}: יתרה פתוחה {self.open_balance} גדולה מסכום החשבונית {self.amount}"
            )
        return self

    @property
    def open_money(self) -> Money:
        return Money.of(self.open_balance, self.currency)

    def to_invoice(self) -> Invoice:
        """אותה חשבונית בלי שדה היתרה."""
        return Invoice(**self.model_dump(exclude={"open_balance"}))


class BankTransaction(_Record):
    """תנועת בנק אחת מהקובץ היומי. מקבילה בעולם האמיתי: שורה בייצוא היומי של Nilus."""

    tx_id: Identifier
    tx_date: date = Field(alias="date")
    account_id: Identifier
    amount: Decimal  # יכול להיות שלילי (תנועה יוצאת) - הסינון מחליט מה לעשות איתה
    currency: Currency
    bank_text: str = ""  # לפעמים הבנק לא מעביר שום טקסט

    @model_validator(mode="after")
    def _valid_amount(self) -> BankTransaction:
        if self.amount == 0:
            raise ValueError(f"{self.tx_id}: תנועה בסכום אפס")
        Money.of(self.amount, self.currency)
        return self

    @property
    def money(self) -> Money:
        return Money.of(self.amount, self.currency)

    @property
    def is_incoming(self) -> bool:
        return self.amount > 0


class Alias(_Record):
    """כינוי: השם כפי שמופיע בבנק -> לקוח. הזיכרון שהסוכן לומד מאישורים."""

    bank_name: NonEmptyText
    customer_id: Identifier
    times_confirmed: Annotated[int, Field(ge=0)] = 0


class FxRate(_Record):
    """שער יציג של בנק ישראל: כמה שקלים שווה יחידה אחת של מטבע זר ביום מסוים."""

    rate_date: date = Field(alias="date")
    currency: Currency
    rate_to_ils: Annotated[Decimal, Field(gt=0)]

    @model_validator(mode="after")
    def _foreign_only(self) -> FxRate:
        if self.currency == Currency.ILS:
            raise ValueError("אין שער לשקל מול שקל")
        return self
