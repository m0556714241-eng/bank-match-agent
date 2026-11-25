"""
בדיקות שמירת המצב (ה-71): האובייקטים שלנו חוזרים מהקובץ בדיוק כמו שנשמרו,
בלי אזהרת "סוג לא רשום", וגם אחרי "הפעלה מחדש" (חיבור חדש לאותו קובץ).
"""

import logging
import warnings
from datetime import date
from decimal import Decimal
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.adapters.interfaces import Adjustment, AdjustmentType, Application, Certainty
from src.domain.models import BankTransaction
from src.domain.money import Money
from src.engine.results import Candidate, CustomerMatch, CustomerOutcome, InvoiceMatch, MatchKind
from src.graph.checkpoint import make_serializer, open_checkpointer
from src.llm.customer_llm import LlmCustomerSuggestion
from src.llm.explain import ExplanationResult, RunTotals

FX_MATCH = InvoiceMatch(
    certainty=Certainty.PROBABLE,
    customer_id="C005",
    kind=MatchKind.FX,
    receipt_amount=Money.of("2200", "USD"),
    applications=(Application("INV-10026", Money.of("2200", "USD")),),
    adjustments=(Adjustment(AdjustmentType.FX_DIFFERENCE, Money.of("61.60", "ILS")),),
    notes=("הערה",),
    evidence={"invoice": "INV-10026"},
)

OBJECTS = [
    Money.of("7037.80", "ILS"),
    BankTransaction(tx_id="TX-1", date=date(2026, 11, 16), account_id="ACC-01", amount=Decimal("7037.80"),
                    currency="ILS", bank_text="test"),
    FX_MATCH,
    CustomerMatch(CustomerOutcome.NEEDS_LLM, candidates=(Candidate("C001", 91.0),), invoice_refs=("INV-1",)),
    LlmCustomerSuggestion("C002", "השם תואם"),
    RunTotals(5, 1, 1, 1, 2, (Money.of("10", "ILS"),)),
    ExplanationResult({"TX-1": "משפט"}, "סיכום", ("TX-2",)),
]


def test_round_trip_is_identical_without_warnings(caplog):
    serde = make_serializer()
    caplog.set_level(logging.WARNING)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for obj in OBJECTS:
            back = serde.loads_typed(serde.dumps_typed(obj))
            assert back == obj, type(obj).__name__  # כולל tuple שנשאר tuple
    messages = caplog.text + " ".join(str(w.message) for w in caught)
    assert "unregistered" not in messages  # כל המחלקות ברשימת ההיתר


def test_state_survives_restart(tmp_path):
    class State(TypedDict):
        match: object

    def node(state: State) -> dict:
        return {"match": FX_MATCH}

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    config = {"configurable": {"thread_id": "run-test-1"}}

    first = open_checkpointer(tmp_path / "checkpoints.db")
    builder.compile(checkpointer=first).invoke({"match": None}, config, durability="sync")
    first.conn.close()  # "סגירת התוכנה"

    second = open_checkpointer(tmp_path / "checkpoints.db")  # "הפעלה מחדש" - חיבור חדש לאותו קובץ
    restored = builder.compile(checkpointer=second).get_state(config).values["match"]
    second.conn.close()
    assert restored == FX_MATCH
    assert isinstance(restored.applications, tuple)
