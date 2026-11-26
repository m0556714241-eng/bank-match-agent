"""
בדיקות הגרף (שלב 6; ה-72 סעיף 3, ה-74 סעיף 7, ה-75, ה-76).

כל בדיקה כאן היא בדיקה שהתשובה עליה ידועה מראש:
- נתוני בנק אמיתיים מהפרויקט, מסד זמני (איפוס לתיקייה זמנית), שומר זמני.
- מודל מדומה בלבד: עונה תשובה קבועה, או "נופל" בכוונה. אין שום קריאה למודל אמיתי (ה-64).
- יום 08/11 הוא היום הראשון בטווח, ולכן על מסד טרי התוצאה שלו זהה לקובץ התשובות.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from scripts.reset_db import reset
from src.adapters.boi_fx_provider import BoiFxProvider
from src.adapters.csv_bank_source import EXPECTED_COLUMNS, CsvBankSource
from src.adapters.interfaces import Certainty
from src.config import load_settings
from src.engine.policy import MatchPolicy
from src.engine.results import ExceptionReason
from src.graph.build import RunNotWaitingError, build_run_graph, make_run_id, resume_run, run_config, start_run
from src.graph.checkpoint import open_checkpointer
from src.graph.run_registry import RunBusyError, RunRegistry, RunStatus
from src.graph.state import ApprovalAction, Decision
from src.graph.toolbox import Toolbox
from src.llm.customer_llm import MODEL_UNAVAILABLE_REASON

SETTINGS = load_settings()  # נתיבי קבצי הקלט האמיתיים של הפרויקט
DAY_ONE = date(2026, 11, 8)  # היום הראשון: 6 העברות, 3 מהן תקבולים
DAY_WITH_LLM = date(2026, 11, 10)  # יום שבו TX-00020 ("בטא") צריכה את המודל


# ---------- מודלים מדומים ----------


class FakeModel:
    """מודל מדומה: ממלא את הטופס בתשובה קבועה, וסופר קריאות."""

    def __init__(self, answer: dict):
        self.answer = answer  # התשובה הקבועה
        self.calls = 0  # כמה פעמים קראו לו

    def with_structured_output(self, schema):
        fake = self

        class _Runner:
            def invoke(self, messages):
                fake.calls += 1
                return schema(**fake.answer)  # Pydantic בודק את התשובה כמו מול מודל אמיתי

        return _Runner()


class ModelOutage(Exception):
    """תקלה מדומה במודל (כמו ניתוק / זמן המתנה). מחלקה רגילה - ולכן RetryPolicy מנסה שוב."""


class FailingModel:
    """מודל שתמיד נופל - לבדיקת הניסיונות החוזרים ותוכנית החירום."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        fake = self

        class _Runner:
            def invoke(self, messages):
                fake.calls += 1
                raise ModelOutage("המודל לא זמין (מדומה)")

        return _Runner()


UNKNOWN_ANSWER = {"customer_id": "UNKNOWN", "reason": "לא ברור", "quote": "", "matched_name": ""}  # זיהוי: "לא יודע"
EMPTY_EXPLAIN = {"lines": [], "summary": ""}  # הסברים: ריק -> הכל חוזר לתבניות (צפוי ויציב)


# ---------- סביבת בדיקה ----------


@pytest.fixture
def env(tmp_path):
    """מסד זמני, שומר זמני, טבלת ריצות זמנית. נסגר בסוף הבדיקה."""
    ledger = reset(tmp_path / "ledger.db", SETTINGS.customers_csv, SETTINGS.invoices_csv, SETTINGS.aliases_csv)
    checkpointer = open_checkpointer(tmp_path / "checkpoints.db")
    registry = RunRegistry(tmp_path / "checkpoints.db")  # באותו קובץ, כמו בהפעלה האמיתית
    resources = {"tmp": tmp_path, "ledger": ledger, "checkpointer": checkpointer, "registry": registry}
    yield resources
    resources["checkpointer"].conn.close()  # סוגר את השומר הנוכחי (בבדיקת ה"קריסה" - את החדש)


def _toolbox(ledger, bank_dir: Path = SETTINGS.bank_dir, identify=None, explain=None) -> Toolbox:
    """ארגז כלים לבדיקה: מסד זמני + מודלים מדומים."""
    return Toolbox(
        ledger=ledger,
        bank=CsvBankSource(bank_dir),
        fx=BoiFxProvider(SETTINGS.fx_cache_csv, lookback_days=SETTINGS.fx_lookback_days),
        policy=MatchPolicy(invoice_number_pattern=SETTINGS.invoice_number_pattern),
        identify_model=identify or FakeModel(UNKNOWN_ANSWER),
        explain_model=explain or FakeModel(EMPTY_EXPLAIN),
    )


def _run_id(day: date) -> str:
    return make_run_id(day, datetime(2026, 11, 26, 22, 30, 0))  # שעה קבועה - מזהה צפוי


def _approve_all(result: dict) -> list[Decision]:
    """אישור כל השורות שמותר לאשר (מה שמנהלת החשבונות הייתה לוחצת)."""
    rows = result["__interrupt__"][0].value["rows"]
    return [Decision(r.tx_id, ApprovalAction.APPROVE, "test") for r in rows if ApprovalAction.APPROVE in r.allowed_actions]


def _by_tx(state: dict) -> dict:
    return {r.tx.tx_id: r for r in state["checked_results"]}


# ---------- 1-2: ריצת יום, כמה העברות במקביל ----------


def test_day_runs_until_approval_with_known_results(env):
    graph = build_run_graph(env["checkpointer"])
    run_id = _run_id(DAY_ONE)
    result = start_run(graph, _toolbox(env["ledger"]), env["registry"], DAY_ONE, run_id)

    assert "__interrupt__" in result  # עצר לאישור
    assert env["registry"].status(run_id) == RunStatus.WAITING
    results = _by_tx(result)
    assert sorted(results) == [f"TX-0000{i}" for i in range(1, 7)]  # 6 העברות במקביל - תוצאה אחת לכל אחת

    # התשובות הידועות מקובץ התשובות (eval/answer_key_draft.csv)
    expected = {
        "TX-00001": (Certainty.CERTAIN, "C001", ["INV-10001"]),
        "TX-00002": (Certainty.CERTAIN, "C002", ["INV-10006", "INV-10007"]),
        "TX-00003": (Certainty.PROBABLE, "C004", ["INV-10014"]),
    }
    for tx_id, (certainty, customer, invoices) in expected.items():
        match = results[tx_id].match
        assert (match.certainty, match.customer_id, [a.invoice_id for a in match.applications]) == (
            certainty, customer, invoices
        ), tx_id
    for tx_id in ("TX-00004", "TX-00005", "TX-00006"):
        assert results[tx_id].match is None  # סוננו - לא תקבול

    rows = result["__interrupt__"][0].value["rows"]
    assert [r.tx_id for r in rows] == ["TX-00001", "TX-00002", "TX-00003"]  # רק תקבולים מוצגים לאישור


# ---------- 3-4, 7: עצירה, חידוש, חידוש כפול, חידוש שנכשל ----------


def test_resume_saves_decisions_and_finishes(env):
    graph = build_run_graph(env["checkpointer"])
    tools, run_id = _toolbox(env["ledger"]), _run_id(DAY_ONE)
    started = start_run(graph, tools, env["registry"], DAY_ONE, run_id)
    decisions = _approve_all(started)

    final = resume_run(graph, tools, env["registry"], run_id, decisions)

    assert final["decisions"] == decisions  # ההחלטות נשמרו בזיכרון הריצה
    assert graph.get_state(run_config(run_id)).next == ()  # הריצה הסתיימה
    assert env["registry"].status(run_id) == RunStatus.DONE


def test_second_resume_is_blocked(env):
    graph = build_run_graph(env["checkpointer"])
    tools, run_id = _toolbox(env["ledger"]), _run_id(DAY_ONE)
    decisions = _approve_all(start_run(graph, tools, env["registry"], DAY_ONE, run_id))
    resume_run(graph, tools, env["registry"], run_id, decisions)

    with pytest.raises(RunBusyError):  # שכבה 1: הנעילה לא נותנת לחדש ריצה שהסתיימה
        resume_run(graph, tools, env["registry"], run_id, decisions)


def test_resume_of_unknown_run_is_blocked(env):
    graph = build_run_graph(env["checkpointer"])
    with pytest.raises(RunBusyError):  # ריצה שלא קיימת - אין מה לתפוס
        resume_run(graph, _toolbox(env["ledger"]), env["registry"], "run-does-not-exist", [])


def test_failed_resume_releases_lock_and_can_retry(env):
    graph = build_run_graph(env["checkpointer"])
    tools, run_id = _toolbox(env["ledger"]), _run_id(DAY_ONE)
    started = start_run(graph, tools, env["registry"], DAY_ONE, run_id)

    bad = [Decision("TX-99999", ApprovalAction.APPROVE, "test")]  # העברה שלא הוצגה - _validate_decisions נכשל
    with pytest.raises(ValueError):
        resume_run(graph, tools, env["registry"], run_id, bad)
    assert env["registry"].status(run_id) == RunStatus.WAITING  # הנעילה שוחררה
    assert "human_approval" in graph.get_state(run_config(run_id)).next  # עדיין מחכה לאישור

    resume_run(graph, tools, env["registry"], run_id, _approve_all(started))  # ניסיון תקין - עובר
    assert env["registry"].status(run_id) == RunStatus.DONE


def test_registry_check_matches_checkpoint(env):
    """שכבה 2: גם אם הטבלה אומרת "מחכה" - קובץ השמירה הוא הקובע."""
    graph = build_run_graph(env["checkpointer"])
    env["registry"].mark_waiting("run-only-in-registry")  # רשומה בטבלה, אבל אין לה נקודת שמירה
    with pytest.raises(RunNotWaitingError):
        resume_run(graph, _toolbox(env["ledger"]), env["registry"], "run-only-in-registry", [])
    assert env["registry"].status("run-only-in-registry") == RunStatus.WAITING  # שוחרר


# ---------- 5-6: הנעילה עצמה ----------


def test_only_one_claim_succeeds(env):
    registry = env["registry"]
    registry.mark_waiting("run-x")
    registry.claim("run-x")  # החלון הראשון תפס
    with pytest.raises(RunBusyError):
        registry.claim("run-x")  # החלון השני נדחה
    other_window = RunRegistry(env["tmp"] / "checkpoints.db")  # חיבור נפרד - כמו תהליך אחר
    with pytest.raises(RunBusyError):
        other_window.claim("run-x")


def test_stale_lock_can_be_claimed(env):
    registry = env["registry"]
    registry.mark_waiting("run-y")
    registry.claim("run-y")
    old = (datetime.now() - timedelta(minutes=16)).isoformat()  # "התוכנית קרסה" לפני 16 דקות
    with sqlite3.connect(env["tmp"] / "checkpoints.db") as conn:
        conn.execute("UPDATE run_registry SET updated_at = ? WHERE run_id = 'run-y'", (old,))
    registry.claim("run-y")  # נעילה נטושה - מותר לתפוס
    assert registry.status("run-y") == RunStatus.RESUMING


# ---------- 8: שתי העברות על אותה חשבונית ----------


def test_two_transfers_for_same_invoice_conflict(env):
    bank_dir = env["tmp"] / "bank"
    bank_dir.mkdir()
    rows = [
        ",".join(EXPECTED_COLUMNS),
        "TX-90001,2026-11-08,ACC-01,4200,ILS,שופ-נט מרקטפלייס בע״מ",
        "TX-90002,2026-11-08,ACC-01,4200,ILS,שופ-נט מרקטפלייס בע״מ",  # אותו סכום, אותו לקוח - אותה חשבונית
    ]
    (bank_dir / "bank_transactions_2026-11-08.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    graph = build_run_graph(env["checkpointer"])
    result = start_run(graph, _toolbox(env["ledger"], bank_dir), env["registry"], DAY_ONE, _run_id(DAY_ONE))

    results = _by_tx(result)
    assert results["TX-90001"].match.certainty == Certainty.CERTAIN  # הראשונה (לפי מזהה) נשארת
    assert [a.invoice_id for a in results["TX-90001"].match.applications] == ["INV-10001"]
    conflict = results["TX-90002"].match
    assert conflict.exception_reason == ExceptionReason.INVOICE_CONFLICT  # השנייה - חריג
    assert conflict.evidence == {"invoice": "INV-10001", "other_tx": "TX-90001"}


# ---------- 9-10: כישלון מודל ----------


def test_identify_model_failure_becomes_exception(env):
    failing = FailingModel()
    graph = build_run_graph(env["checkpointer"])
    tools, run_id = _toolbox(env["ledger"], identify=failing), _run_id(DAY_WITH_LLM)

    verify_runs = 0
    for _namespace, update in graph.stream(
        {"run_id": run_id, "run_date": DAY_WITH_LLM},
        config=run_config(run_id),
        context=tools,
        durability="sync",
        stream_mode="updates",
        subgraphs=True,  # גם עדכונים מתוך תת-הגרף
    ):
        verify_runs += "verify_llm_customer" in update

    assert failing.calls >= 3 and failing.calls % 3 == 0  # 3 ניסיונות לכל העברה שהגיעה למודל
    assert verify_runs == failing.calls // 3  # האימות רץ פעם אחת לכל העברה - לא פעמיים
    tx = _by_tx(graph.get_state(run_config(run_id)).values)["TX-00020"]
    assert tx.match.certainty == Certainty.EXCEPTION  # לעולם לא אישור
    assert tx.match.exception_reason == ExceptionReason.CUSTOMER_NOT_IDENTIFIED
    assert MODEL_UNAVAILABLE_REASON in tx.match.notes  # הערה לאדם: "המודל לא היה זמין"


def test_explain_model_failure_uses_templates_once(env):
    failing = FailingModel()
    graph = build_run_graph(env["checkpointer"])
    tools, run_id = _toolbox(env["ledger"], explain=failing), _run_id(DAY_ONE)

    prepare_runs = 0
    for update in graph.stream(
        {"run_id": run_id, "run_date": DAY_ONE},
        config=run_config(run_id),
        context=tools,
        durability="sync",
        stream_mode="updates",
    ):
        prepare_runs += "prepare_approval" in update

    assert failing.calls == 3  # 3 ניסיונות ואז תוכנית החירום
    assert prepare_runs == 1  # הכנת המסך רצה פעם אחת בלבד
    state = graph.get_state(run_config(run_id))
    assert state.values["explanations"].model_failed  # הכל מתבניות
    assert "human_approval" in state.next  # והריצה עדיין עצרה לאישור כרגיל


# ---------- 11: המשך אחרי "קריסה" ----------


def test_resume_after_restart(env):
    tools, run_id = _toolbox(env["ledger"]), _run_id(DAY_ONE)
    started = start_run(build_run_graph(env["checkpointer"]), tools, env["registry"], DAY_ONE, run_id)
    env["checkpointer"].conn.close()  # "התוכנה נסגרה"

    reopened = open_checkpointer(env["tmp"] / "checkpoints.db")  # "הפעלה מחדש"
    env["checkpointer"] = reopened  # כדי שהסגירה בסוף הבדיקה תסגור את החדש
    graph = build_run_graph(reopened)
    final = resume_run(graph, tools, RunRegistry(env["tmp"] / "checkpoints.db"), run_id, _approve_all(started))

    assert len(final["decisions"]) == 3  # ממשיכים מאותה נקודה, עם אותן שורות
    assert graph.get_state(run_config(run_id)).next == ()
