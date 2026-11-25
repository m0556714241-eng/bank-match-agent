"""
כסף - ייצוג מדויק של סכומים בכל הפרויקט.

הכללים (החלטה ה-31):
- בקוד: כל סכום הוא Money (סכום Decimal + מטבע). אף פעם לא float.
- במסד: מספר שלם ביחידה הקטנה של המטבע (אגורות / סנטים).
- עיגול: רק בקובץ הזה, ובכלל אחד בלבד.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum


class Currency(StrEnum):
    """המטבעות שהסוכן מכיר (החלטה ה-04)."""

    ILS = "ILS"
    USD = "USD"
    EUR = "EUR"


# כמה ספרות אחרי הנקודה יש לכל מטבע (לפי התקן הבינלאומי ISO 4217)
_MINOR_DIGITS: dict[Currency, int] = {
    Currency.ILS: 2,
    Currency.USD: 2,
    Currency.EUR: 2,
}

# כלל העיגול היחיד בפרויקט: חצי ומעלה - מעגלים כלפי מעלה (2.345 -> 2.35)
ROUNDING = ROUND_HALF_UP


class MoneyError(ValueError):
    """שגיאה בפעולה על כסף - למשל חיבור שקלים לדולרים, או סכום לא תקין."""


def parse_decimal(value: str | int | Decimal) -> Decimal:
    """ממיר ערך שהגיע מבחוץ (קובץ, מסד) למספר עשרוני מדויק. דוחה float."""
    # float כבר איבד דיוק לפני שהגיע לכאן - לכן אסור לקבל אותו בכלל
    if isinstance(value, float):
        raise MoneyError(f"float אסור בסכומי כסף: {value!r}. יש להעביר מחרוזת או Decimal.")
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise MoneyError(f"'{value}' אינו מספר תקין") from exc
    # Decimal מסוגל לייצג "אינסוף" ו"לא מספר" - בכסף זה תמיד שגיאה
    if not result.is_finite():
        raise MoneyError(f"'{value}' אינו סכום סופי")
    return result


def to_currency(value: Currency | str) -> Currency:
    """ממיר קוד מטבע (למשל "ILS") לערך של Currency. מטבע לא מוכר = MoneyError."""
    try:
        return Currency(value)
    except ValueError as exc:
        raise MoneyError(f"מטבע לא נתמך: {value!r}") from exc


def _unit(currency: Currency) -> Decimal:
    """היחידה הקטנה של המטבע כמספר עשרוני: 0.01 לשקל, לדולר ולאירו."""
    return Decimal(1).scaleb(-_MINOR_DIGITS[currency])


@dataclass(frozen=True, slots=True)
class Money:
    """סכום כסף במטבע מסוים. בלתי ניתן לשינוי, ולא מאפשר לערבב מטבעות."""

    amount: Decimal
    currency: Currency

    def __post_init__(self) -> None:
        # המרה של מחרוזת כמו "ILS" לערך של Currency (ושגיאה על מטבע לא מוכר)
        currency = to_currency(self.currency)
        if not isinstance(self.amount, Decimal):
            raise MoneyError(f"הסכום חייב להיות Decimal, התקבל {type(self.amount).__name__}")
        unit = _unit(currency)
        # סכום עם יותר ספרות ממה שהמטבע מאפשר (למשל 10.005 ש"ח) הוא כנראה באג -
        # לא מעגלים בשקט. מי שבאמת צריך עיגול קורא במפורש ל-Money.rounded
        if self.amount != self.amount.quantize(unit):
            raise MoneyError(
                f"{self.amount} {currency}: יותר ספרות ממה שהמטבע מאפשר. "
                "לעיגול מכוון יש להשתמש ב-Money.rounded"
            )
        # נרמול לתצוגה אחידה: 10 -> 10.00
        object.__setattr__(self, "amount", self.amount.quantize(unit))
        object.__setattr__(self, "currency", currency)

    # ---------- יצירה ----------

    @classmethod
    def of(cls, value: str | int | Decimal, currency: Currency | str) -> Money:
        """יצירה מערך חיצוני (למשל "4200.00" מקובץ). מדויק בלבד - בלי עיגול."""
        return cls(parse_decimal(value), to_currency(currency))

    @classmethod
    def rounded(cls, value: Decimal, currency: Currency | str) -> Money:
        """יצירה מתוצאת חישוב (למשל סכום כפול שער). המקום היחיד שבו מעגלים."""
        cur = to_currency(currency)
        return cls(value.quantize(_unit(cur), rounding=ROUNDING), cur)

    @classmethod
    def zero(cls, currency: Currency | str) -> Money:
        """אפס במטבע מסוים - נקודת התחלה לסכימה."""
        return cls(Decimal(0), to_currency(currency))

    @classmethod
    def from_minor(cls, minor: int, currency: Currency | str) -> Money:
        """קריאה מהמסד: מספר שלם של אגורות/סנטים -> Money."""
        if isinstance(minor, bool) or not isinstance(minor, int):
            raise MoneyError(f"ביחידה הקטנה מצפים למספר שלם, התקבל {minor!r}")
        cur = to_currency(currency)
        return cls(Decimal(minor).scaleb(-_MINOR_DIGITS[cur]), cur)

    def to_minor(self) -> int:
        """כתיבה למסד: Money -> מספר שלם של אגורות/סנטים."""
        return int(self.amount.scaleb(_MINOR_DIGITS[self.currency]))

    # ---------- חשבון ----------

    def _same_currency(self, other: object) -> Money:
        """בדיקה משותפת לכל פעולה בין שני סכומים: חייבים להיות באותו מטבע."""
        if not isinstance(other, Money):
            raise MoneyError(f"פעולה בין Money לבין {type(other).__name__} אינה מותרת")
        if other.currency != self.currency:
            raise MoneyError(f"אי אפשר לשלב {self.currency} עם {other.currency} בלי המרה")
        return other

    def __add__(self, other: Money) -> Money:
        return Money(self.amount + self._same_currency(other).amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        return Money(self.amount - self._same_currency(other).amount, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def __abs__(self) -> Money:
        return Money(abs(self.amount), self.currency)

    def scaled(self, factor: Decimal) -> Money:
        """הכפלה במקדם (למשל 0.97 לניכוי של 3%). התוצאה מעוגלת לפי הכלל."""
        if isinstance(factor, float):
            raise MoneyError("המקדם חייב להיות Decimal, לא float")
        return Money.rounded(self.amount * factor, self.currency)

    # ---------- השוואה ----------

    def __lt__(self, other: Money) -> bool:
        return self.amount < self._same_currency(other).amount

    def __le__(self, other: Money) -> bool:
        return self.amount <= self._same_currency(other).amount

    def __gt__(self, other: Money) -> bool:
        return self.amount > self._same_currency(other).amount

    def __ge__(self, other: Money) -> bool:
        return self.amount >= self._same_currency(other).amount

    def is_zero(self) -> bool:
        return self.amount == 0

    def is_positive(self) -> bool:
        return self.amount > 0

    def __str__(self) -> str:
        return f"{self.amount:,.{_MINOR_DIGITS[self.currency]}f} {self.currency}"


def convert(money: Money, rate: Decimal, to: Currency | str) -> Money:
    """המרת מטבע: סכום × שער, מעוגל במטבע היעד. השער עצמו נשמר בדיוק מלא."""
    if isinstance(rate, float) or not isinstance(rate, Decimal):
        raise MoneyError("השער חייב להיות Decimal")
    if rate <= 0:
        raise MoneyError(f"שער חייב להיות חיובי: {rate}")
    return Money.rounded(money.amount * rate, to)
