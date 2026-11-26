"""
איפוס המסד: מוחק אותו ובונה אותו מחדש מקבצי הקלט (החלטה ה-30).

מתי מריצים: לפני ריצה ראשונה, ולפני כל הרצת הערכה - כדי שכל ריצה תתחיל מאותה נקודה.
רק בסביבת בדיקות (APP_ENV=development). במערכת אמיתית (production) האיפוס חסום -
בדומה לרענון סביבת הבדיקות (Sandbox) ב-NetSuite, שלא נוגע בחשבון האמיתי.
הפעלה מתיקיית הפרויקט:
    python -m scripts.reset_db          (שואל לפני מחיקה)
    python -m scripts.reset_db --yes    (בלי שאלה - להרצות אוטומטיות)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.adapters.sqlite_ledger import LedgerSeedError, SqliteLedger
from src.config import load_settings
from src.domain.models import Alias, Customer, OpenInvoice

T = TypeVar("T", bound=BaseModel)


class SeedFileError(Exception):
    """קובץ קלט פגום - האיפוס נעצר לפני שנגעו במסד."""


def read_records(path: Path, model: type[T]) -> list[T]:
    """קורא קובץ CSV ומחזיר רשימת רשומות בדוקות. שורה פגומה אחת = עצירה עם הסבר."""
    if not path.exists():
        raise SeedFileError(f"הקובץ לא נמצא: {path}")
    records: list[T] = []
    problems: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            try:
                records.append(model.model_validate(row))
            except ValidationError as exc:
                err = exc.errors()[0]
                problems.append(f"  שורה {row_number}: {'.'.join(map(str, err['loc']))} - {err['msg']}")
    if problems:
        raise SeedFileError(f"{path.name} - שורות פגומות:\n" + "\n".join(problems))
    return records


def reset(
    db_path: Path,
    customers_csv: Path,
    invoices_csv: Path,
    aliases_csv: Path,
    checkpoint_path: Path | None = None,  # קובץ השמירה של הגרף (רשות) - נמחק יחד עם המסד (ה-71)
) -> SqliteLedger:
    """קורא ובודק את כל הקבצים, ורק אחר כך מוחק ובונה את המסד מחדש (וגם מוחק ריצות שעצרו לאישור)."""
    # קודם קוראים הכול: אם קובץ פגום - המסד הישן וקובץ השמירה נשארים כמו שהם
    customers = read_records(customers_csv, Customer)
    invoices = read_records(invoices_csv, OpenInvoice)
    aliases = read_records(aliases_csv, Alias)

    if db_path.exists():
        db_path.unlink()
    if checkpoint_path is not None:
        _delete_checkpoints(checkpoint_path)  # ריצה ישנה שעצרה לא תתאים למסד החדש
    return SqliteLedger.create(db_path, customers, invoices, aliases)


def _delete_checkpoints(path: Path) -> None:
    """עזר של reset: מוחק את קובץ השמירה ואת שני קבצי הלוואי ש-SQLite יוצר לידו."""
    for suffix in ("", "-wal", "-shm"):  # הקובץ עצמו + קבצי הלוואי של מצב WAL
        Path(f"{path}{suffix}").unlink(missing_ok=True)  # לא קיים - לא שגיאה


def main() -> int:
    """נקודת הכניסה מהטרמינל. מחזיר 0 בהצלחה, 1 בכישלון (המוסכמה של שורת פקודה)."""
    parser = argparse.ArgumentParser(description="איפוס מסד הנהלת החשבונות מקבצי הקלט")
    parser.add_argument("--yes", action="store_true", help="מחיקה בלי שאלה")
    args = parser.parse_args()

    settings = load_settings()

    # במערכת אמיתית אסור לאפס - אף פעם, גם לא עם --yes
    if settings.is_production:
        print("האיפוס חסום: APP_ENV=production (מערכת אמיתית). המסד לא השתנה.")
        return 1

    # המסד מכיל רישומים - לא מוחקים בלי אישור מפורש
    if settings.db_path.exists() and not args.yes:
        answer = input(
            f"המסד {settings.db_path.name} יימחק וכל הרישומים בו, וגם הריצות שעצרו לאישור. להמשיך? (yes/no): "
        )
        if answer.strip().lower() != "yes":
            print("בוטל. המסד לא השתנה.")
            return 1

    try:
        ledger = reset(
            settings.db_path,
            settings.customers_csv,
            settings.invoices_csv,
            settings.aliases_csv,
            checkpoint_path=settings.checkpoint_path,  # גם קובץ השמירה של הגרף (ה-71)
        )
    except (SeedFileError, LedgerSeedError) as exc:
        print(f"האיפוס נכשל:\n{exc}")
        return 1

    print("המסד נבנה מחדש:")
    print(f"  לקוחות:  {len(ledger.customers())}")
    print(f"  כינויים: {len(ledger.aliases())}")
    print(f"  מיקום:   {settings.db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
