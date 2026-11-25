"""
בדיקת המנוע מול קובץ התשובות (דאטהסט א', 80 תנועות) - בלי מודל השפה ובלי גרף.

מדמה את הימים לפי הסדר, על מסד זמני: כל הצעה שאינה חריג "מאושרת" ונרשמת,
כדי שהיתרות ישתנו כמו במציאות (למשל תשלום כפול ביום מאוחר יותר).
תנועה שצריכה מודל: משתמשים בלקוח מקובץ התשובות במקום המודל ("מודל מושלם"),
כדי לבדוק את מה שקורה אחריו. כך נמדד המנוע עצמו.

הפעלה מתיקיית הפרויקט:
    python -m scripts.evaluate_engine
"""

from __future__ import annotations

import csv
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from scripts.reset_db import reset
from src.adapters.boi_fx_provider import BoiFxProvider
from src.adapters.csv_bank_source import CsvBankSource
from src.adapters.interfaces import Certainty, ReceiptPosting
from src.adapters.sqlite_ledger import SqliteLedger
from src.config import PROJECT_ROOT, load_settings
from src.engine.customer_id import identify_customer, verify_llm_customer
from src.engine.filters import filter_tx
from src.engine.invoice_match import find_invoice
from src.engine.policy import MatchPolicy
from src.engine.results import CustomerOutcome, FilterOutcome

ANSWER_KEY = PROJECT_ROOT / "eval" / "answer_key_draft.csv"

# המילים בקובץ התשובות -> רמות הוודאות שלנו
KEY_CERTAINTY = {"certain": Certainty.CERTAIN, "reasonable": Certainty.PROBABLE, "exception": Certainty.EXCEPTION}


@dataclass
class Outcome:
    """התוצאה של תנועה אחת, בפורמט שמשווים לקובץ התשובות."""

    certainty: str  # certain / reasonable / exception / filter
    customer: str
    invoices: str
    reason: str
    used_llm: bool


def main() -> int:
    settings = load_settings()
    policy = MatchPolicy(invoice_number_pattern=settings.invoice_number_pattern)
    key = _read_key()

    with tempfile.TemporaryDirectory() as tmp:
        ledger = reset(Path(tmp) / "eval.db", settings.customers_csv, settings.invoices_csv, settings.aliases_csv)
        bank = CsvBankSource(settings.bank_dir)
        fx = BoiFxProvider(settings.fx_cache_csv, lookback_days=settings.fx_lookback_days)
        customers = ledger.customers()

        results: dict[str, Outcome] = {}
        day = date(2026, 11, 8)
        while day <= date(2026, 11, 21):
            for tx in bank.load_day(day).transactions:
                results[tx.tx_id] = _run_one(tx, ledger, fx, policy, customers, key)
            day += timedelta(days=1)

    return _report(results, key)


def _run_one(tx, ledger, fx, policy, customers, key) -> Outcome:
    """עזר של main: מריץ את צמתי המנוע על תנועה אחת, ורושם אם יצא הצעה."""
    f = filter_tx(tx, policy)
    if f.outcome == FilterOutcome.FILTERED:
        return Outcome("filter", "", "", str(f.filter_reason), False)
    if f.outcome == FilterOutcome.UNSUPPORTED:
        return Outcome("exception", "", "", "UNSUPPORTED_TX_TYPE", False)

    open_now = ledger.open_invoices(tx.tx_date)
    paid = ledger.paid_invoices(tx.tx_date)

    cm = identify_customer(tx, customers, ledger.aliases(), open_now, policy)
    used_llm = False
    if cm.outcome == CustomerOutcome.NEEDS_LLM:
        used_llm = True
        oracle = key[tx.tx_id]["expected_customer"] or None  # "מודל מושלם" מקובץ התשובות
        cm = verify_llm_customer(oracle, customers, cm.invoice_refs)
        if cm.outcome != CustomerOutcome.IDENTIFIED:
            return Outcome("exception", "", "", "CUSTOMER_NOT_IDENTIFIED", True)

    m = find_invoice(tx, cm, open_now, customers, fx, policy, paid)
    invoices = ";".join(a.invoice_id for a in m.applications)
    if not m.is_exception:
        ledger.post_receipt(ReceiptPosting(
            tx_id=tx.tx_id, customer_id=m.customer_id, amount=m.receipt_amount, certainty=m.certainty,
            approved_by="eval", run_id="eval", applications=list(m.applications), adjustments=list(m.adjustments),
        ))
    certainty = {Certainty.CERTAIN: "certain", Certainty.PROBABLE: "reasonable", Certainty.EXCEPTION: "exception"}[m.certainty]
    return Outcome(certainty, m.customer_id or "", invoices if not m.is_exception else "", str(m.exception_reason or ""), used_llm)


def _read_key() -> dict[str, dict[str, str]]:
    with ANSWER_KEY.open(encoding="utf-8-sig", newline="") as f:
        return {r["tx_id"]: r for r in csv.DictReader(f)}


def _report(results: dict[str, Outcome], key: dict[str, dict[str, str]]) -> int:
    """עזר של main: משווה לקובץ התשובות ומדפיס את הפערים."""
    wrong_approvals, mismatches = [], []
    for tx_id, exp in key.items():
        got = results.get(tx_id)
        if got is None:
            mismatches.append((tx_id, "לא נקלטה", ""))
            continue
        exp_cert = "filter" if exp["case_type"] == "filter" else exp["expected_certainty"]
        exp_reason = exp["note"].split(" ")[0] if exp_cert == "exception" else ""
        proposed = got.certainty in ("certain", "reasonable")
        right_target = got.customer == exp["expected_customer"] and got.invoices == exp["expected_invoices"]
        if proposed and (exp_cert not in ("certain", "reasonable") or not right_target):
            wrong_approvals.append(tx_id)  # הצעה שהייתה מאושרת - אבל שגויה (המדד הראשי)
        ok = got.certainty == exp_cert
        if ok and proposed:
            ok = right_target
        elif ok and exp_cert == "exception":
            ok = exp_reason == "" or got.reason == exp_reason.rstrip(",:")
        if not ok:
            mismatches.append((tx_id, f"צפוי: {exp_cert} {exp['expected_customer']} {exp['expected_invoices']} {exp_reason}",
                               f"התקבל: {got.certainty} {got.customer} {got.invoices} {got.reason}"))

    counts = Counter(r.certainty for r in results.values())
    print(f"תנועות: {len(results)}  |  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"הגיעו למודל: {sum(r.used_llm for r in results.values())}")
    print(f"התאמה לקובץ התשובות: {len(key) - len(mismatches)}/{len(key)}")
    print(f"אישורים שגויים (מדד ראשי, מטרה 0): {len(wrong_approvals)}")
    for tx_id, exp, got in mismatches:
        print(f"  {tx_id}  {exp}\n           {got}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
