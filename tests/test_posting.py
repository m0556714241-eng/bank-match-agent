"""
בדיקות הרישום (שלב 7, ה-78). כמו בבדיקות הגרף: נתונים אמיתיים, מסד ושומר זמניים, מודל מדומה,
וכל תשובה ידועה מראש. כל בדיקה עוברת את כל הדרך: ריצה -> עצירה לאישור -> החלטות -> רישום.
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal

import pytest

from src.adapters.csv_bank_source import EXPECTED_COLUMNS
from src.adapters.interfaces import Application, Certainty, PostingControlError, ReceiptPosting
from src.adapters.sqlite_ledger import SqliteLedger
from src.domain.models import BankTransaction
from src.domain.money import Money
from src.engine.customer_id import identify_customer
from src.engine.policy import MatchPolicy
from src.graph import nodes
from src.graph.build import build_run_graph, continue_posting, resume_run, run_config, start_run
from src.graph.state import ApprovalAction, Decision, PostingStatus
from src.graph.run_registry import RunStatus
from tests.test_graph import DAY_ONE, SETTINGS, _approve_all, _run_id, _toolbox, env  # noqa: F401 (env = fixture)


# ---------- עזר ----------


def _start(env, ledger=None, bank_dir=None):
    """מריץ את יום 08/11 עד העצירה לאישור. מחזיר: גרף, ארגז כלים, מזהה ריצה, התוצאה."""
    graph = build_run_graph(env["checkpointer"])
    tools = _toolbox(ledger or env["ledger"], **({"bank_dir": bank_dir} if bank_dir else {}))  # ארגז כלים לבדיקה
    run_id = _run_id(DAY_ONE)
    started = start_run(graph, tools, env["registry"], DAY_ONE, run_id)
    return graph, tools, run_id, started


def _postings(state: dict) -> dict:
    return {p.tx_id: p for p in state["postings"]}


def _open(ledger, invoice_id: str) -> Money:
    return {i.invoice_id: i.open_money for i in ledger.open_invoices(DAY_ONE)}.get(invoice_id, Money.of("0", "ILS"))


class ControlFailingLedger(SqliteLedger):
    """מסד שבדיקת הבקרה שלו נכשלת בהעברה אחת - כישלון עסקי נקודתי."""

    def post_receipt(self, posting):
        if posting.tx_id == "TX-00002":
            raise PostingControlError("בקרה מדומה נכשלה")
        return super().post_receipt(posting)


class BrokenLedger(SqliteLedger):
    """מסד שנופל בתקלה מערכתית בהעברה אחת - כמו מסד נעול."""

    def post_receipt(self, posting):
        if posting.tx_id == "TX-00002":
            raise sqlite3.OperationalError("database is locked (מדומה)")
        return super().post_receipt(posting)


# ---------- אישור / דחייה / בלי החלטה ----------


def test_approve_posts_reduces_balance_and_audits(env):
    graph, tools, run_id, started = _start(env)
    final = resume_run(graph, tools, env["registry"], run_id, _approve_all(started))

    postings = _postings(final)
    assert {p.status for p in postings.values()} == {PostingStatus.POSTED}  # 3 תקבולים נרשמו
    assert _open(env["ledger"], "INV-10001").is_zero()  # היתרה ירדה עד 0
    audit = env["ledger"].audit_entries(run_id)
    assert [(a.tx_id, a.action, a.outcome) for a in audit] == [
        ("TX-00001", "APPROVE", "POSTED"), ("TX-00002", "APPROVE", "POSTED"), ("TX-00003", "APPROVE", "POSTED"),
    ]
    assert audit[0].receipt_id == postings["TX-00001"].receipt_id
    assert '"customer"' in audit[0].proposal  # ההצעה שהוצגה נשמרה


def test_reject_goes_to_manual_and_is_not_posted(env):
    graph, tools, run_id, _ = _start(env)
    final = resume_run(graph, tools, env["registry"], run_id, [Decision("TX-00001", ApprovalAction.REJECT, "test")])

    postings = _postings(final)
    assert postings["TX-00001"].status == PostingStatus.MANUAL
    assert env["ledger"].receipt_for_tx("TX-00001") is None  # לא נרשם כסף
    assert "TX-00001" in env["ledger"].processed_tx_ids(["TX-00001"])  # לא תוצע שוב
    assert env["ledger"].audit_entries(run_id)[0].action == "REJECT"


def test_no_decision_stays_pending(env):
    graph, tools, run_id, _ = _start(env)
    final = resume_run(graph, tools, env["registry"], run_id, [Decision("TX-00001", ApprovalAction.APPROVE, "test")])

    postings = _postings(final)
    assert postings["TX-00002"].status == PostingStatus.PENDING
    assert env["ledger"].processed_tx_ids(["TX-00002"]) == set()  # תוצע שוב בריצה הבאה
    assert [a.tx_id for a in env["ledger"].audit_entries(run_id)] == ["TX-00001"]  # אין יומן בלי החלטה


# ---------- בקרה לפני רישום ----------


def test_balance_changed_since_shown_is_not_posted(env):
    graph, tools, run_id, started = _start(env)
    env["ledger"].post_receipt(ReceiptPosting(  # מישהו רשם 100 ש"ח על החשבונית אחרי שהוצגה במסך
        tx_id="TX-OTHER", customer_id="C001", amount=Money.of("100", "ILS"), certainty=Certainty.CERTAIN,
        approved_by="someone", run_id="other", applications=[Application("INV-10001", Money.of("100", "ILS"))],
    ))
    final = resume_run(graph, tools, env["registry"], run_id, _approve_all(started))

    postings = _postings(final)
    assert postings["TX-00001"].status == PostingStatus.CHANGED_SINCE_SHOWN  # לא נרשם על יתרה אחרת
    assert env["ledger"].receipt_for_tx("TX-00001") is None
    assert postings["TX-00002"].status == PostingStatus.POSTED  # השאר ממשיכות


# ---------- "כבר נרשמה" ----------


def test_already_posted_identical_is_skipped(env):
    graph, tools, run_id, started = _start(env)
    env["ledger"].post_receipt(ReceiptPosting(  # בדיוק אותו רישום כבר קיים (כמו אחרי קריסה באמצע)
        tx_id="TX-00001", customer_id="C001", amount=Money.of("4200", "ILS"), certainty=Certainty.CERTAIN,
        approved_by="test", run_id=run_id, applications=[Application("INV-10001", Money.of("4200", "ILS"))],
    ))
    final = resume_run(graph, tools, env["registry"], run_id, _approve_all(started))

    assert _postings(final)["TX-00001"].status == PostingStatus.ALREADY_POSTED
    assert sqlite3.connect(env["tmp"] / "ledger.db").execute(
        "SELECT COUNT(*) FROM receipts WHERE tx_id = 'TX-00001'").fetchone()[0] == 1  # תקבול אחד בלבד


def test_already_posted_different_is_conflict(env):
    graph, tools, run_id, started = _start(env)
    other = next(i for i in env["ledger"].open_invoices(DAY_ONE, "C001") if i.invoice_id != "INV-10001")
    env["ledger"].post_receipt(ReceiptPosting(  # רשום להעברה הזו תקבול אחר לגמרי
        tx_id="TX-00001", customer_id="C001", amount=Money.of("10", "ILS"), certainty=Certainty.PROBABLE,
        approved_by="someone", run_id="other", applications=[Application(other.invoice_id, Money.of("10", "ILS"))],
    ))
    final = resume_run(graph, tools, env["registry"], run_id, _approve_all(started))

    assert _postings(final)["TX-00001"].status == PostingStatus.CONFLICTS_WITH_EXISTING  # לטיפול ידני


# ---------- כישלון נקודתי מול תקלה מערכתית ----------


def test_control_failure_in_one_transfer_does_not_stop_others(env):
    ledger = ControlFailingLedger(env["tmp"] / "ledger.db")
    graph, tools, run_id, started = _start(env, ledger)
    final = resume_run(graph, tools, env["registry"], run_id, _approve_all(started))

    postings = _postings(final)
    assert postings["TX-00002"].status == PostingStatus.CONTROL_FAILED
    assert postings["TX-00001"].status == PostingStatus.POSTED
    assert postings["TX-00003"].status == PostingStatus.POSTED


def test_system_error_stops_posting_and_continue_finishes(env):
    graph, broken_tools, run_id, started = _start(env, BrokenLedger(env["tmp"] / "ledger.db"))
    with pytest.raises(sqlite3.OperationalError):  # תקלה מערכתית - הרישום נעצר
        resume_run(graph, broken_tools, env["registry"], run_id, _approve_all(started))
    assert graph.get_state(run_config(run_id)).next == ("post_receipts",)  # נעצר באמצע הרישום
    assert env["registry"].status(run_id) == RunStatus.WAITING

    final = continue_posting(graph, _toolbox(env["ledger"]), env["registry"], run_id)  # התקלה נפתרה - ממשיכים

    postings = _postings(final)
    assert postings["TX-00001"].status == PostingStatus.ALREADY_POSTED  # נרשמה לפני התקלה - לא שוב
    assert postings["TX-00002"].status == PostingStatus.POSTED
    assert env["registry"].status(run_id) == RunStatus.DONE


# ---------- החלה על הוותיקה ----------


def _partial_day(env):
    """יום עם העברה חלקית בלי מספר חשבונית: 37 ש"ח מ"מטבחי השרון בע״מ"."""
    bank_dir = env["tmp"] / "bank"
    bank_dir.mkdir()
    rows = [",".join(EXPECTED_COLUMNS), "TX-90010,2026-11-08,ACC-01,37,ILS,מטבחי השרון בע״מ"]
    (bank_dir / "bank_transactions_2026-11-08.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return bank_dir


def test_apply_to_oldest_shows_target_and_posts_it(env):
    graph, tools, run_id, started = _start(env, bank_dir=_partial_day(env))
    [row] = started["__interrupt__"][0].value["rows"]
    oldest = min(env["ledger"].open_invoices(DAY_ONE, "C010"), key=lambda i: (i.issue_date, i.invoice_id))

    assert row.allowed_actions == (ApprovalAction.APPLY_TO_OLDEST, ApprovalAction.REJECT)
    assert row.lines[0].invoice_id == oldest.invoice_id  # היעד מוצג מראש במסך
    before = oldest.open_money

    final = resume_run(graph, tools, env["registry"], run_id,
                       [Decision("TX-90010", ApprovalAction.APPLY_TO_OLDEST, "test")])

    assert _postings(final)["TX-90010"].status == PostingStatus.POSTED
    assert _open(env["ledger"], oldest.invoice_id) == before - Money.of("37", "ILS")


def test_apply_to_oldest_target_changed_is_not_posted(env):
    graph, tools, run_id, started = _start(env, bank_dir=_partial_day(env))
    oldest = started["__interrupt__"][0].value["rows"][0].lines[0]
    env["ledger"].post_receipt(ReceiptPosting(  # החשבונית הוותיקה נסגרה בינתיים
        tx_id="TX-CLOSE", customer_id="C010", amount=oldest.open_before, certainty=Certainty.CERTAIN,
        approved_by="someone", run_id="other", applications=[Application(oldest.invoice_id, oldest.open_before)],
    ))
    final = resume_run(graph, tools, env["registry"], run_id,
                       [Decision("TX-90010", ApprovalAction.APPLY_TO_OLDEST, "test")])

    assert _postings(final)["TX-90010"].status == PostingStatus.CHANGED_SINCE_SHOWN  # לא עובר בשקט לחשבונית אחרת
    assert env["ledger"].receipt_for_tx("TX-90010") is None


# ---------- כינויים ----------


def _times_confirmed(ledger, name: str, customer_id: str) -> int:
    return next((a.times_confirmed for a in ledger.aliases() if a.bank_name == name and a.customer_id == customer_id), 0)


def test_alias_is_learned_after_approval(env):
    before = _times_confirmed(env["ledger"], "סופר בית רשתות", "C004")  # TX-00003 זוהתה דרך כינוי
    graph, tools, run_id, started = _start(env)
    final = resume_run(graph, tools, env["registry"], run_id, _approve_all(started))

    assert _times_confirmed(env["ledger"], "סופר בית רשתות", "C004") == before + 1
    assert "כינוי נלמד" in _postings(final)["TX-00003"].detail
    assert _postings(final)["TX-00001"].detail == ""  # TX-00001 זוהתה לפי שם זהה - אין מה ללמוד


def test_conflicting_alias_is_not_learned(env):
    graph, tools, run_id, started = _start(env)
    result = next(r for r in started["checked_results"] if r.tx.tx_id == "TX-00003")  # "סופר בית רשתות" = C004
    note = nodes._learn_alias(result, "C001", env["ledger"])  # מישהו מנסה לשייך את השם ללקוח אחר

    assert note.startswith("כינוי לא נלמד")
    assert _times_confirmed(env["ledger"], "סופר בית רשתות", "C001") == 0


def test_alias_never_gives_certain(env):
    ledger = env["ledger"]
    for _ in range(5):  # גם אחרי 5 אישורים
        ledger.confirm_alias("סופר בית רשתות", "C004")
    tx = BankTransaction(tx_id="TX-T", date=DAY_ONE, account_id="ACC-01", amount=Decimal("7760"),
                         currency="ILS", bank_text="סופר בית רשתות")
    match = identify_customer(tx, ledger.customers(), ledger.aliases(), ledger.open_invoices(DAY_ONE),
                              MatchPolicy(invoice_number_pattern=SETTINGS.invoice_number_pattern))
    assert match.customer_ids == ("C004",) and not match.strong  # לכל היותר "סביר"
