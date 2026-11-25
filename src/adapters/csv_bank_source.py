"""
מתאם תנועות בנק מקבצי CSV יומיים - מימוש של הממשק BankSource.

זה שער הכניסה של הנתונים: כל שורה נבדקת כאן, ושורה פגומה לא נקלטת
אלא נרשמת כשגיאת קליטה עם קוד סיבה (החלטות ה-39, ה-41). שאר הקובץ ממשיך.
בעולם האמיתי: מתאם שקורא את הייצוא היומי של Nilus.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from src.adapters.interfaces import (
    BankLoadResult,
    IntakeError,
    IntakeErrorCode,
    IntakeWarning,
    IntakeWarningCode,
)
from src.domain.models import BankTransaction
from src.domain.money import Currency, Money, MoneyError, parse_decimal

# העמודות שחייבות להופיע בקובץ, בדיוק אלה
EXPECTED_COLUMNS = ["tx_id", "date", "account_id", "amount", "currency", "bank_text"]

# שדות שאסור שיהיו ריקים (טקסט הבנק מותר להיות ריק)
REQUIRED_FIELDS = ["tx_id", "date", "account_id", "amount", "currency"]

# "רעש" = הפרש של פחות מאלפית אגורה מהסכום המעוגל (למשל שארית של Excel)
NOISE_TOLERANCE = Decimal("0.00001")

# כמה ימים אחורה תאריך תנועה יכול להיות ביחס ליום הקובץ, בלי אזהרה
# (העברה משישי/שבת/חג נרשמת בבנק ביום העסקים הבא). יעבור לקובץ ההגדרות.
DEFAULT_DATE_TOLERANCE_DAYS = 4


class _RowProblem(Exception):
    """בעיה בשורה אחת - נתפסת בתוך הלולאה והופכת לשגיאת קליטה."""

    def __init__(self, code: IntakeErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class CsvBankSource:
    """קורא קובץ בנק אחד לכל יום מתוך תיקייה."""

    def __init__(
        self,
        folder: Path,
        file_pattern: str = "bank_transactions_{day}.csv",
        date_tolerance_days: int = DEFAULT_DATE_TOLERANCE_DAYS,
    ) -> None:
        self._folder = folder
        self._file_pattern = file_pattern
        self._date_tolerance = timedelta(days=date_tolerance_days)

    def load_day(self, day: date) -> BankLoadResult:
        """כל התנועות של יום אחד, אחרי בדיקה. שורות פגומות - לרשימת שגיאות הקליטה."""
        path = self._folder / self._file_pattern.format(day=day.isoformat())

        # אין קובץ ליום הזה - לא קורסים, אבל מדווחים (בעולם האמיתי: הייצוא לא הגיע)
        if not path.exists():
            warning = IntakeWarning(path.name, None, IntakeWarningCode.FILE_MISSING, f"לא הגיע קובץ בנק ליום {day}")
            return BankLoadResult(transactions=[], warnings=[warning])

        # utf-8-sig מתעלם מהסימן הנסתר (BOM) ש-Excel מוסיף בתחילת הקובץ
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []

            # מבנה הקובץ השתנה - לא מנחשים: שום שורה לא נקלטת
            if header != EXPECTED_COLUMNS:
                return BankLoadResult(
                    transactions=[],
                    intake_errors=[
                        IntakeError(
                            source=path.name,
                            row_number=1,
                            code=IntakeErrorCode.MISSING_FIELD,
                            message=f"מבנה העמודות שונה מהצפוי. התקבל: {header}",
                            raw={},
                        )
                    ],
                )

            transactions: list[BankTransaction] = []
            errors: list[IntakeError] = []
            warnings: list[IntakeWarning] = []
            seen_ids: set[str] = set()

            # שורה 1 היא הכותרת, ולכן הנתונים מתחילים בשורה 2
            for row_number, raw in enumerate(reader, start=2):
                try:
                    tx, row_warnings = self._parse_row(raw, day, seen_ids)
                except _RowProblem as problem:
                    errors.append(IntakeError(path.name, row_number, problem.code, problem.message, dict(raw)))
                    continue
                transactions.append(tx)
                seen_ids.add(tx.tx_id)
                # לכל אזהרה של השורה מצרפים את שם הקובץ ומספר השורה
                for code, message in row_warnings:
                    warnings.append(IntakeWarning(path.name, row_number, code, message))

        return BankLoadResult(transactions=transactions, intake_errors=errors, warnings=warnings)

    # ---------- עזר ----------

    def _parse_row(
        self, raw: dict[str, str], day: date, seen_ids: set[str]
    ) -> tuple[BankTransaction, list[tuple[IntakeWarningCode, str]]]:
        """עזר של load_day: בודק שורה אחת ומחזיר תנועה + אזהרות (סוג, הסבר). בעיה = _RowProblem."""
        row = {k: (v or "").strip() for k, v in raw.items()}

        for name in REQUIRED_FIELDS:
            if not row[name]:
                raise _RowProblem(IntakeErrorCode.MISSING_FIELD, f"השדה {name} ריק")

        if row["tx_id"] in seen_ids:
            raise _RowProblem(IntakeErrorCode.DUPLICATE_TX_ID, f"{row['tx_id']} מופיעה פעמיים בקובץ")

        if row["currency"] not in Currency.__members__:
            raise _RowProblem(IntakeErrorCode.UNKNOWN_CURRENCY, f"מטבע לא נתמך: {row['currency']}")

        try:
            tx_day = date.fromisoformat(row["date"])
        except ValueError:
            raise _RowProblem(IntakeErrorCode.INVALID_DATE, f"תאריך לא תקין: {row['date']}") from None

        row_warnings: list[tuple[IntakeWarningCode, str]] = []

        amount, rounded_note = _normalize_amount(row["amount"], Currency(row["currency"]))
        row["amount"] = str(amount)
        if rounded_note:
            row_warnings.append((IntakeWarningCode.AMOUNT_ROUNDED, rounded_note))

        try:
            tx = BankTransaction.model_validate(row)
        except ValidationError as exc:
            raise _RowProblem(IntakeErrorCode.INVALID_AMOUNT, _first_error(exc)) from None

        # תאריך התנועה ביחס ליום הקובץ: עד כמה ימים אחורה - רגיל (סוף שבוע, חג)
        if tx_day > day:
            row_warnings.append(
                (IntakeWarningCode.DATE_OUT_OF_RANGE,
                 f"תאריך עתידי {tx_day} בקובץ של {day} - ייתכן שהקובץ שויך ליום הלא נכון")
            )
        elif day - tx_day > self._date_tolerance:
            row_warnings.append(
                (IntakeWarningCode.DATE_OUT_OF_RANGE,
                 f"תנועה מ-{tx_day} בקובץ של {day} - האם כבר נקלטה בקובץ אחר?")
            )

        return tx, row_warnings


def _normalize_amount(raw: str, currency: Currency) -> tuple[Decimal, str | None]:
    """עזר של _parse_row: מנרמל סכום לפי ה-39. מחזיר סכום + אזהרה, או זורק _RowProblem."""
    try:
        value = parse_decimal(raw)
    except MoneyError:
        raise _RowProblem(IntakeErrorCode.INVALID_AMOUNT, f"סכום לא תקין: {raw}") from None

    rounded = Money.rounded(value, currency).amount
    diff = abs(value - rounded)

    if diff == 0:
        return rounded, None  # תקין (כולל אפסים מיותרים כמו 4200.000)
    if diff < NOISE_TOLERANCE:
        return rounded, f"סכום {raw} עוגל ל-{rounded} (רעש עשרוני)"
    raise _RowProblem(
        IntakeErrorCode.INVALID_AMOUNT_PRECISION,
        f"סכום {raw} כולל יותר ספרות ממה שהמטבע {currency} מאפשר",
    )


def _first_error(exc: ValidationError) -> str:
    """עזר של _parse_row: הודעה קצרה מהשגיאה הראשונה של pydantic."""
    err = exc.errors()[0]
    field = ".".join(str(p) for p in err["loc"]) or "שורה"
    return f"{field}: {err['msg']}"
