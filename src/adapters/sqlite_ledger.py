"""
"מערכת הנהלת החשבונות" + הזיכרון של הסוכן, במסד SQLite - מימוש של הממשק LedgerStore.

עקרונות (החלטות ה-30, ה-31, ה-33):
- כסף נשמר במסד כמספר שלם ביחידה הקטנה, ובקוד הוא תמיד Money.
- רשומות כספיות רק נוספות. היתרה מחושבת בתצוגה open_balances.
- רישום תקבול = יחידה אחת: הכול נשמר, או ששום דבר לא נשמר.
בעולם האמיתי: מתאם שכותב ל-NetSuite (דרך Workato).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

from src.adapters.interfaces import (
    DuplicatePostingError,
    PostingControlError,
    PostingResult,
    ReceiptPosting,
)
from src.domain.models import Alias, Customer, Invoice, OpenInvoice
from src.domain.money import Money

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class LedgerSeedError(Exception):
    """הנתונים ההתחלתיים לא עקביים (למשל חשבונית של לקוח שלא קיים). המסד לא נוצר."""


class SqliteLedger:
    """גישה למסד. כל פונקציה פותחת חיבור משלה וסוגרת אותו בסוף."""

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"המסד לא קיים: {db_path}. יש ליצור אותו עם SqliteLedger.create")
        self._db_path = db_path

    # ---------- יצירה וטעינה ראשונית ----------

    @classmethod
    def create(
        cls,
        db_path: Path,
        customers: list[Customer],
        invoices: list[OpenInvoice],
        aliases: list[Alias],
    ) -> SqliteLedger:
        """יוצר מסד חדש מהקבצים: לקוחות, חשבוניות (+ יתרות פתיחה) וכינויים."""
        if db_path.exists():
            raise FileExistsError(f"המסד כבר קיים: {db_path}. לאיפוס יש להריץ את reset_db")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = _open(db_path)
        try:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            with conn:  # טרנזקציה אחת: או שהכול נטען, או ששום דבר
                _seed(conn, customers, invoices, aliases)
        except sqlite3.IntegrityError as exc:
            conn.close()
            db_path.unlink()  # לא משאירים מסד חצי-טעון
            raise LedgerSeedError(f"הנתונים ההתחלתיים לא עקביים: {exc}") from exc
        conn.close()
        return cls(db_path)

    # ---------- קריאה ----------

    def customers(self) -> list[Customer]:
        """כל הלקוחות."""
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM customers ORDER BY customer_id").fetchall()
        return [Customer.model_validate(dict(r)) for r in rows]

    def open_invoices(self, as_of: date, customer_id: str | None = None) -> list[OpenInvoice]:
        """חשבוניות שלא שולמו במלואן, שהופקו עד התאריך. אפשר לסנן לפי לקוח."""
        sql = "SELECT * FROM open_balances WHERE status <> 'PAID' AND issue_date <= ?"
        params: list[str] = [as_of.isoformat()]
        if customer_id is not None:
            sql += " AND customer_id = ?"
            params.append(customer_id)
        with self._read() as conn:
            rows = conn.execute(sql + " ORDER BY issue_date, invoice_id", params).fetchall()
        return [_row_to_open_invoice(r) for r in rows]

    def paid_invoices(self, as_of: date, customer_id: str | None = None) -> list[Invoice]:
        """חשבוניות ששולמו במלואן, שהופקו עד התאריך - לזיהוי תשלום כפול."""
        sql = "SELECT * FROM open_balances WHERE status = 'PAID' AND issue_date <= ?"
        params: list[str] = [as_of.isoformat()]
        if customer_id is not None:
            sql += " AND customer_id = ?"
            params.append(customer_id)
        with self._read() as conn:
            rows = conn.execute(sql + " ORDER BY issue_date, invoice_id", params).fetchall()
        return [_row_to_open_invoice(r).to_invoice() for r in rows]

    def aliases(self) -> list[Alias]:
        """כל הכינויים הידועים."""
        with self._read() as conn:
            rows = conn.execute("SELECT * FROM aliases ORDER BY bank_name, customer_id").fetchall()
        return [Alias.model_validate(dict(r)) for r in rows]

    def processed_tx_ids(self, tx_ids: list[str]) -> set[str]:
        """מתוך הרשימה - אילו תנועות כבר טופלו. שאלה אחת למסד לכל הרשימה."""
        if not tx_ids:
            return set()
        marks = ",".join("?" * len(tx_ids))
        with self._read() as conn:
            rows = conn.execute(f"SELECT tx_id FROM processed_bank_tx WHERE tx_id IN ({marks})", tx_ids)
            return {r["tx_id"] for r in rows}

    # ---------- כתיבה ----------

    def confirm_alias(self, bank_name: str, customer_id: str) -> Alias:
        """אישור כינוי: חדש - נוסף עם מונה 1; קיים - המונה עולה ב-1."""
        conn = _open(self._db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO aliases (bank_name, customer_id, times_confirmed) VALUES (?, ?, 1)
                    ON CONFLICT (bank_name, customer_id)
                    DO UPDATE SET times_confirmed = times_confirmed + 1
                    """,
                    (bank_name, customer_id),
                )
            row = conn.execute(
                "SELECT * FROM aliases WHERE bank_name = ? AND customer_id = ?", (bank_name, customer_id)
            ).fetchone()
        finally:
            conn.close()
        return Alias.model_validate(dict(row))

    def mark_processed(self, tx_id: str, run_id: str, result: str) -> None:
        """סימון תנועה שטופלה בלי תקבול (FILTERED / REJECTED)."""
        conn = _open(self._db_path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO processed_bank_tx (tx_id, run_id, result, processed_at) VALUES (?, ?, ?, ?)",
                    (tx_id, run_id, result, _now()),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicatePostingError(f"{tx_id} כבר סומנה כמטופלת") from exc
        finally:
            conn.close()

    def post_receipt(self, posting: ReceiptPosting) -> PostingResult:
        """רישום תקבול + החלות + הפרשים + סימון כמטופלת - הכול או כלום, כולל בדיקת בקרה."""
        planned = _planned_per_invoice(posting)  # כמה אמור לרדת מכל חשבונית
        conn = _open(self._db_path)
        try:
            # IMMEDIATE = נועלים את המסד לכתיבה כבר עכשיו, כדי שאף אחד לא ישנה יתרות באמצע
            conn.execute("BEGIN IMMEDIATE")

            before = _balances(conn, list(planned))
            _check_invoices_belong_to_customer(conn, posting.customer_id, list(planned))

            receipt_id = _next_receipt_id(conn)
            _insert_receipt(conn, receipt_id, posting)
            for app in posting.applications:
                conn.execute(
                    "INSERT INTO receipt_applications (receipt_id, invoice_id, applied_minor, currency, source)"
                    " VALUES (?, ?, ?, ?, 'AGENT')",
                    (receipt_id, app.invoice_id, app.amount.to_minor(), app.amount.currency),
                )
            for adj in posting.adjustments:
                conn.execute(
                    "INSERT INTO adjustments (receipt_id, type, amount_minor, currency) VALUES (?, ?, ?, ?)",
                    (receipt_id, adj.type, adj.amount.to_minor(), adj.amount.currency),
                )
            conn.execute(
                "INSERT INTO processed_bank_tx (tx_id, run_id, result, processed_at) VALUES (?, ?, 'POSTED', ?)",
                (posting.tx_id, posting.run_id, _now()),
            )

            # בדיקת בקרה (סעיף 11): קריאה חוזרת - כל יתרה ירדה בדיוק כמתוכנן?
            after = _balances(conn, list(planned))
            for invoice_id, amount in planned.items():
                if before[invoice_id] - after[invoice_id] != amount:
                    raise PostingControlError(
                        f"{invoice_id}: היתרה ירדה ב-{before[invoice_id] - after[invoice_id]} במקום {amount}"
                    )

            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            if "UNIQUE" in str(exc) or "PRIMARY KEY" in str(exc):
                raise DuplicatePostingError(f"{posting.tx_id} כבר נרשמה או טופלה") from exc
            raise PostingControlError(f"{posting.tx_id}: המסד סירב לרישום - {exc}") from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return PostingResult(receipt_id=receipt_id, balances_after=after)

    # ---------- עזר פנימי למחלקה ----------

    def _read(self) -> sqlite3.Connection:
        """עזר לפונקציות הקריאה: חיבור שנסגר אוטומטית בסוף הבלוק with."""
        return _ClosingConnection(self._db_path)


# ====================================================================
# פונקציות עזר
# ====================================================================


def _open(db_path: Path) -> sqlite3.Connection:
    """עזר לכל החיבורים: פותח חיבור עם ההגדרות הנכונות."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # גישה לעמודות לפי שם: row["customer_id"]
    conn.execute("PRAGMA foreign_keys = ON")  # ב-SQLite בדיקת REFERENCES כבויה כברירת מחדל
    return conn


class _ClosingConnection:
    """עזר של _read: מאפשר לכתוב with ומבטיח שהחיבור נסגר."""

    def __init__(self, db_path: Path) -> None:
        self._conn = _open(db_path)

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, *exc: object) -> None:
        self._conn.close()


def _now() -> str:
    """עזר: התאריך והשעה עכשיו, בשעון אחיד (UTC), כטקסט."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _seed(
    conn: sqlite3.Connection, customers: list[Customer], invoices: list[OpenInvoice], aliases: list[Alias]
) -> None:
    """עזר של create: טוען לקוחות, חשבוניות, יתרות פתיחה וכינויים."""
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?)",
        [(c.customer_id, c.name, c.currency, str(c.withholding_rate)) for c in customers],
    )
    conn.executemany(
        "INSERT INTO invoices VALUES (?, ?, ?, ?, ?)",
        [(i.invoice_id, i.customer_id, i.issue_date.isoformat(), i.money.to_minor(), i.currency) for i in invoices],
    )
    # חשבונית ששולמה בחלקה לפני התקופה (כמו INV-10042): החלה אחת של "יתרת פתיחה"
    for inv in invoices:
        already_paid = inv.money - inv.open_money
        if already_paid.is_positive():
            conn.execute(
                "INSERT INTO receipt_applications (receipt_id, invoice_id, applied_minor, currency, source)"
                " VALUES (NULL, ?, ?, ?, 'OPENING_BALANCE')",
                (inv.invoice_id, already_paid.to_minor(), inv.currency),
            )
    conn.executemany(
        "INSERT INTO aliases VALUES (?, ?, ?)",
        [(a.bank_name, a.customer_id, a.times_confirmed) for a in aliases],
    )


def _row_to_open_invoice(row: sqlite3.Row) -> OpenInvoice:
    """עזר של open_invoices: שורה מהמסד (אגורות) -> OpenInvoice (Decimal)."""
    return OpenInvoice(
        invoice_id=row["invoice_id"],
        customer_id=row["customer_id"],
        issue_date=date.fromisoformat(row["issue_date"]),
        amount=Money.from_minor(row["amount_minor"], row["currency"]).amount,
        currency=row["currency"],
        open_balance=Money.from_minor(row["open_minor"], row["currency"]).amount,
    )


def _planned_per_invoice(posting: ReceiptPosting) -> dict[str, Money]:
    """עזר של post_receipt: כמה אמור לרדת מכל חשבונית (אותה חשבונית יכולה להופיע פעמיים)."""
    planned: dict[str, Money] = defaultdict(lambda: Money.zero(posting.amount.currency))
    for app in posting.applications:
        planned[app.invoice_id] = planned[app.invoice_id] + app.amount
    return dict(planned)


def _balances(conn: sqlite3.Connection, invoice_ids: list[str]) -> dict[str, Money]:
    """עזר של post_receipt: היתרות הפתוחות כרגע של רשימת חשבוניות."""
    marks = ",".join("?" * len(invoice_ids))
    rows = conn.execute(
        f"SELECT invoice_id, open_minor, currency FROM open_balances WHERE invoice_id IN ({marks})", invoice_ids
    ).fetchall()
    found = {r["invoice_id"]: Money.from_minor(r["open_minor"], r["currency"]) for r in rows}
    missing = set(invoice_ids) - set(found)
    if missing:
        raise PostingControlError(f"חשבוניות לא קיימות: {sorted(missing)}")
    return found


def _check_invoices_belong_to_customer(conn: sqlite3.Connection, customer_id: str, invoice_ids: list[str]) -> None:
    """עזר של post_receipt: כל החשבוניות שייכות ללקוח של התקבול - אחרת תשלום נרשם ללקוח הלא נכון."""
    marks = ",".join("?" * len(invoice_ids))
    rows = conn.execute(
        f"SELECT invoice_id FROM invoices WHERE invoice_id IN ({marks}) AND customer_id <> ?",
        [*invoice_ids, customer_id],
    ).fetchall()
    if rows:
        raise PostingControlError(f"חשבוניות של לקוח אחר: {[r['invoice_id'] for r in rows]}")


def _next_receipt_id(conn: sqlite3.Connection) -> str:
    """עזר של post_receipt: מספר התקבול הבא, R-000001 וכן הלאה."""
    count = conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
    return f"R-{count + 1:06d}"


def _insert_receipt(conn: sqlite3.Connection, receipt_id: str, posting: ReceiptPosting) -> None:
    """עזר של post_receipt: שורת התקבול עצמה."""
    conn.execute(
        "INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            receipt_id,
            posting.tx_id,
            posting.customer_id,
            posting.amount.to_minor(),
            posting.amount.currency,
            posting.certainty,
            posting.approved_by,
            _now(),
            posting.run_id,
        ),
    )
