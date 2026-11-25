"""בדיקות למתאם המסד: טעינה, יתרת פתיחה, רישום כיחידה אחת, בקרה והגנות."""

from datetime import date
from decimal import Decimal

import pytest

from src.adapters.interfaces import (
    Adjustment,
    AdjustmentType,
    Application,
    Certainty,
    DuplicatePostingError,
    PostingControlError,
    ReceiptPosting,
)
from src.adapters.sqlite_ledger import LedgerSeedError, SqliteLedger
from src.domain.models import Alias, Customer, OpenInvoice
from src.domain.money import Money

ILS = "ILS"
TODAY = date(2026, 11, 17)


def _customer(cid, rate="0"):
    return Customer(customer_id=cid, name=f"לקוח {cid}", currency=ILS, withholding_rate=Decimal(rate))


def _invoice(iid, cid, amount, open_balance=None, issued="2026-10-01"):
    return OpenInvoice(
        invoice_id=iid, customer_id=cid, issue_date=date.fromisoformat(issued),
        amount=Decimal(amount), currency=ILS,
        open_balance=Decimal(open_balance if open_balance is not None else amount),
    )


@pytest.fixture
def ledger(tmp_path):
    # מסד קטן: מטבחי השרון (C010) + סופר-בית (C004)
    return SqliteLedger.create(
        tmp_path / "ledger.db",
        customers=[_customer("C010"), _customer("C004", "0.03")],
        invoices=[
            _invoice("INV-10050", "C010", "5000.00"),
            _invoice("INV-10042", "C010", "2200.00", open_balance="1100.00", issued="2026-09-17"),
            _invoice("INV-10020", "C004", "10000.00"),
            _invoice("INV-10099", "C004", "800.00", issued="2026-11-20"),  # מופקת אחרי TODAY
        ],
        aliases=[Alias(bank_name="סופר בית רשתות", customer_id="C004")],
    )


def _posting(tx, cid, amount, apps, adjustments=()):
    return ReceiptPosting(
        tx_id=tx, customer_id=cid, amount=Money.of(amount, ILS), certainty=Certainty.PROBABLE,
        approved_by="miri", run_id="run-1",
        applications=[Application(i, Money.of(a, ILS)) for i, a in apps],
        adjustments=list(adjustments),
    )


def _open(ledger, iid):
    return {i.invoice_id: i.open_money for i in ledger.open_invoices(date(2026, 12, 31))}.get(iid)


# ---------- טעינה ----------


def test_opening_balance(ledger):
    assert _open(ledger, "INV-10042") == Money.of("1100", ILS)


def test_open_invoices_respects_issue_date(ledger):
    ids = [i.invoice_id for i in ledger.open_invoices(TODAY, "C004")]
    assert ids == ["INV-10020"]  # INV-10099 עוד לא הופקה ביום הזה


def test_seed_unknown_customer_fails_and_leaves_no_db(tmp_path):
    path = tmp_path / "bad.db"
    with pytest.raises(LedgerSeedError):
        SqliteLedger.create(path, [_customer("C010")], [_invoice("INV-1", "C999", "10")], [])
    assert not path.exists()


def test_create_refuses_existing_db(ledger, tmp_path):
    with pytest.raises(FileExistsError):
        SqliteLedger.create(tmp_path / "ledger.db", [], [], [])


# ---------- רישום ----------


def test_partial_then_full_payment(ledger):
    r1 = ledger.post_receipt(_posting("TX-00031", "C010", "2000", [("INV-10050", "2000")]))
    assert r1.balances_after["INV-10050"] == Money.of("3000", ILS)
    r2 = ledger.post_receipt(_posting("TX-00058", "C010", "3000", [("INV-10050", "3000")]))
    assert r2.balances_after["INV-10050"] == Money.of("0", ILS)
    assert _open(ledger, "INV-10050") is None  # שולמה - כבר לא ברשימת הפתוחות
    assert r2.receipt_id == "R-000002"


def test_withholding_closes_invoice(ledger):
    # 10,000 חשבונית, 9,700 בבנק: החלה על הסכום המלא + הפרש ניכוי במקור
    adj = Adjustment(AdjustmentType.WITHHOLDING_TAX, Money.of("300", ILS))
    result = ledger.post_receipt(_posting("TX-00042", "C004", "10000", [("INV-10020", "10000")], [adj]))
    assert result.balances_after["INV-10020"] == Money.of("0", ILS)


def test_one_receipt_many_invoices(ledger):
    p = _posting("TX-00060", "C010", "6100", [("INV-10050", "5000"), ("INV-10042", "1100")])
    result = ledger.post_receipt(p)
    assert all(m.is_zero() for m in result.balances_after.values())


# ---------- הגנות ----------


def test_duplicate_tx_blocked(ledger):
    ledger.post_receipt(_posting("TX-00031", "C010", "2000", [("INV-10050", "2000")]))
    with pytest.raises(DuplicatePostingError):
        ledger.post_receipt(_posting("TX-00031", "C010", "1000", [("INV-10050", "1000")]))
    assert _open(ledger, "INV-10050") == Money.of("3000", ILS)  # הנסיון השני לא שינה כלום


def test_over_apply_rolls_back_everything(ledger):
    p = _posting("TX-00070", "C010", "6000", [("INV-10042", "1100"), ("INV-10050", "4900"), ])
    ledger.post_receipt(p)  # תקין
    bad = _posting("TX-00071", "C010", "200", [("INV-10050", "200")])  # נשארו רק 100
    with pytest.raises(PostingControlError):
        ledger.post_receipt(bad)
    # הכול או כלום: התנועה לא סומנה כמטופלת, והיתרה לא השתנתה
    assert ledger.processed_tx_ids(["TX-00071"]) == set()
    assert _open(ledger, "INV-10050") == Money.of("100", ILS)


def test_invoice_of_other_customer_blocked(ledger):
    with pytest.raises(PostingControlError):
        ledger.post_receipt(_posting("TX-00080", "C010", "10000", [("INV-10020", "10000")]))
    assert _open(ledger, "INV-10020") == Money.of("10000", ILS)


def test_unknown_invoice_blocked(ledger):
    with pytest.raises(PostingControlError):
        ledger.post_receipt(_posting("TX-00081", "C010", "10", [("INV-99999", "10")]))


# ---------- זיכרון ----------


def test_processed_and_mark(ledger):
    ledger.post_receipt(_posting("TX-00031", "C010", "2000", [("INV-10050", "2000")]))
    ledger.mark_processed("TX-00043", "run-1", "FILTERED")
    got = ledger.processed_tx_ids(["TX-00031", "TX-00043", "TX-00044"])
    assert got == {"TX-00031", "TX-00043"}
    with pytest.raises(DuplicatePostingError):
        ledger.mark_processed("TX-00043", "run-2", "FILTERED")


def test_confirm_alias_counts(ledger):
    assert ledger.confirm_alias("סופר בית רשתות", "C004").times_confirmed == 1
    assert ledger.confirm_alias("סופר בית רשתות", "C004").times_confirmed == 2
    assert ledger.confirm_alias("קבוצת פלוס", "C010").times_confirmed == 1  # כינוי חדש


def test_paid_invoices(ledger):
    assert ledger.paid_invoices(TODAY) == []
    ledger.post_receipt(_posting("TX-00031", "C010", "5000", [("INV-10050", "5000")]))
    assert [i.invoice_id for i in ledger.paid_invoices(TODAY, "C010")] == ["INV-10050"]
