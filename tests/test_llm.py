"""
בדיקות שלב 5 – מודל השפה. אף בדיקה לא קוראת למודל אמיתי (ה-64):
FakeModel מחזיר תשובה קבועה מראש, ורושם מה נשלח אליו.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.adapters.interfaces import Adjustment, AdjustmentType, Application, Certainty
from src.config import Settings
from src.domain.models import Alias, BankTransaction, Customer
from src.domain.money import Money
from src.engine.policy import MatchPolicy
from src.engine.results import Candidate, ExceptionReason, InvoiceMatch, MatchKind
from src.llm.customer_llm import (
    MODEL_UNAVAILABLE_REASON,
    NO_EVIDENCE_REASON,
    UNKNOWN,
    build_answer_schema,
    build_messages,
    fallback_suggestion,
    suggest_customer,
)
from src.llm.explain import (
    EXCEPTION_TEMPLATES,
    KIND_TEMPLATES,
    SUMMARY_KEY,
    ExplainItem,
    compute_totals,
    explain_run,
    extract_numbers,
    fallback_explanations,
    template_sentence,
    template_summary,
)
from src.llm.model import MissingApiKeyError, make_chat_model

POLICY = MatchPolicy()


# ---------- מודל מדומה ----------


class FakeModel:
    """מודל מדומה: ממלא את הטופס שקיבל בתשובה קבועה, ורושם את ההודעות ואת מספר הקריאות."""

    def __init__(self, answer: dict):
        self.answer = answer  # התשובה שהמודל "יחזיר"
        self.calls = 0  # כמה פעמים קראו לו
        self.messages = None  # מה נשלח אליו בפעם האחרונה

    def with_structured_output(self, schema):
        fake = self

        class _Runner:
            def invoke(self, messages):
                fake.calls += 1
                fake.messages = messages
                return schema(**fake.answer)  # Pydantic בודק את התשובה כמו מול מודל אמיתי

        return _Runner()


# ---------- נתונים לבדיקות ----------


def _tx(tx_id: str, amount: str, currency: str = "ILS", text: str = "העברה") -> BankTransaction:
    return BankTransaction(
        tx_id=tx_id, date=date(2026, 11, 16), account_id="ACC-01", amount=Decimal(amount), currency=currency, bank_text=text
    )


CUSTOMERS = [
    Customer(customer_id="C001", name="בטא-טק בע״מ", currency="ILS", withholding_rate=Decimal("0")),
    Customer(customer_id="C002", name="בטא שיווק", currency="ILS", withholding_rate=Decimal("0")),
]
ALIASES = [Alias(bank_name="BETA MKT", customer_id="C002")]
CANDIDATES = (Candidate("C001", 91.0), Candidate("C002", 93.0))

FX_PROPOSAL = InvoiceMatch(
    certainty=Certainty.PROBABLE,
    customer_id="C005",
    kind=MatchKind.FX,
    receipt_amount=Money.of("2200", "USD"),
    applications=(Application("INV-10026", Money.of("2200", "USD")),),
    adjustments=(Adjustment(AdjustmentType.FX_DIFFERENCE, Money.of("61.60", "ILS")),),
    evidence={"invoice": "INV-10026", "invoice_amount": "2,200.00 USD", "fx_diff": "61.60 ILS", "fx_diff_pct": "0.88%"},
)
PARTIAL_EXCEPTION = InvoiceMatch(
    certainty=Certainty.EXCEPTION, customer_id="C002", exception_reason=ExceptionReason.PARTIAL_WITHOUT_REFERENCE
)
FX_EXCEPTION = InvoiceMatch(
    certainty=Certainty.EXCEPTION,
    customer_id="C005",
    kind=MatchKind.FX,
    exception_reason=ExceptionReason.FX_DIFF_ABOVE_TOLERANCE,
    evidence={"invoice": "INV-10031", "fx_diff": "211.00 ILS", "fx_diff_pct": "6.53%"},
)

ITEMS = [
    ExplainItem(_tx("TX-1", "7037.80"), FX_PROPOSAL, "Acme"),
    ExplainItem(_tx("TX-2", "500"), PARTIAL_EXCEPTION, "בטא שיווק"),
]


def _answer(lines: dict[str, str], summary: str = "טופלו 5 תנועות; 1 חריג דורש בירור.") -> dict:
    return {"lines": [{"tx_id": k, "sentence": v} for k, v in lines.items()], "summary": summary}


# ---------- model.py ----------


def test_make_chat_model_without_key_raises():
    settings = Settings(
        app_env="development", bank_dir=Path("."), customers_csv=Path("."), invoices_csv=Path("."),
        aliases_csv=Path("."), fx_cache_csv=Path("."), db_path=Path("."), invoice_number_pattern=r"INV-\d{5}",
        date_tolerance_days=4, fx_lookback_days=10, anthropic_model="claude-sonnet-5-5",
        anthropic_api_key=None, langsmith_project="test",
    )
    with pytest.raises(MissingApiKeyError):
        make_chat_model(settings, max_tokens=300)


# ---------- customer_llm.py ----------


def test_answer_schema_allows_only_candidates_or_unknown():
    schema = build_answer_schema(("C001", "C002"))
    assert schema(customer_id="C002", reason="x", quote="", matched_name="").customer_id == "C002"
    assert schema(customer_id=UNKNOWN, reason="x", quote="", matched_name="").customer_id == UNKNOWN
    with pytest.raises(ValidationError):  # לקוח שלא ברשימה נדחה
        schema(customer_id="C999", reason="x", quote="", matched_name="")


def test_build_messages_lists_candidates_with_aliases():
    messages = build_messages(_tx("TX-9", "100", text="העב' מבטא שיווק 4471"), CANDIDATES, CUSTOMERS, ALIASES)
    human = messages[1][1]
    assert messages[0][0] == "system"
    assert "העב' מבטא שיווק 4471" in human
    assert "C001 | official name: בטא-טק בע״מ" in human
    assert "BETA MKT" in human  # הכינוי נשלח כרמז


def test_build_messages_marks_empty_bank_text():
    human = build_messages(_tx("TX-9", "100", text=""), CANDIDATES, CUSTOMERS, [])[1][1]
    assert "Bank text: (empty)" in human


TX_BETA = _tx("TX-9", "100", text="העב' מבטא שיווק 4471")


def _choice(customer_id: str, quote: str = "", matched_name: str = "") -> dict:
    return {"customer_id": customer_id, "reason": "השם תואם", "quote": quote, "matched_name": matched_name}


def test_suggest_customer_with_valid_evidence():
    model = FakeModel(_choice("C002", quote="מבטא שיווק", matched_name="בטא שיווק"))
    result = suggest_customer(model, TX_BETA, CANDIDATES, CUSTOMERS, ALIASES, POLICY)
    assert result.customer_id == "C002"
    assert result.quote == "מבטא שיווק"  # יוצג במסך
    assert not result.model_failed and not result.evidence_rejected


def test_evidence_matching_an_alias_is_valid():
    model = FakeModel(_choice("C002", quote="BETA MKT", matched_name="BETA MKT"))
    tx = _tx("TX-9", "100", text="TRF BETA MKT LTD")
    assert suggest_customer(model, tx, CANDIDATES, CUSTOMERS, ALIASES, POLICY).customer_id == "C002"


def test_invented_quote_is_rejected():
    model = FakeModel(_choice("C002", quote="בטא שיווק והפקות", matched_name="בטא שיווק"))  # לא בטקסט
    result = suggest_customer(model, TX_BETA, CANDIDATES, CUSTOMERS, ALIASES, POLICY)
    assert result.customer_id is None
    assert result.evidence_rejected and result.reason == NO_EVIDENCE_REASON


def test_name_of_other_customer_is_rejected():
    model = FakeModel(_choice("C001", quote="מבטא שיווק", matched_name="בטא שיווק"))  # השם של C002
    assert suggest_customer(model, TX_BETA, CANDIDATES, CUSTOMERS, ALIASES, POLICY).customer_id is None


def test_choice_without_evidence_is_rejected():
    model = FakeModel(_choice("C002"))  # בחר לקוח בלי ציטוט
    assert suggest_customer(model, TX_BETA, CANDIDATES, CUSTOMERS, ALIASES, POLICY).customer_id is None


def test_quote_not_similar_to_name_is_rejected():
    model = FakeModel(_choice("C002", quote="4471", matched_name="בטא שיווק"))  # בטקסט, אבל לא שם
    assert suggest_customer(model, TX_BETA, CANDIDATES, CUSTOMERS, ALIASES, POLICY).customer_id is None


def test_suggest_customer_unknown_becomes_none():
    model = FakeModel(_choice(UNKNOWN))
    result = suggest_customer(model, TX_BETA, CANDIDATES, CUSTOMERS, ALIASES, POLICY)
    assert result.customer_id is None and not result.evidence_rejected


def test_suggest_customer_without_candidates_does_not_call_model():
    model = FakeModel(_choice(UNKNOWN))
    result = suggest_customer(model, _tx("TX-9", "100"), (), CUSTOMERS, ALIASES, POLICY)
    assert result.customer_id is None
    assert model.calls == 0


def test_fallback_suggestion_marks_failure():
    result = fallback_suggestion()
    assert result.customer_id is None
    assert result.model_failed
    assert result.reason == MODEL_UNAVAILABLE_REASON


# ---------- explain.py: תבניות וספירה ----------


def test_every_kind_and_exception_has_template():
    assert set(KIND_TEMPLATES) == set(MatchKind)
    assert set(EXCEPTION_TEMPLATES) == set(ExceptionReason)


def test_template_sentence_fills_facts():
    sentence = template_sentence(ITEMS[0])
    assert "INV-10026" in sentence and "61.60" in sentence and "0.88%" in sentence


def test_template_mentions_possible_duplicate():
    dup = InvoiceMatch(
        certainty=Certainty.EXCEPTION,
        customer_id="C002",
        exception_reason=ExceptionReason.OVERPAYMENT,
        evidence={"possible_duplicate_of": "INV-10011"},
    )
    assert "ייתכן תשלום כפול של INV-10011" in template_sentence(ExplainItem(_tx("TX-3", "900"), dup, "בטא שיווק"))


def test_compute_totals_counts_and_sums_per_currency():
    items = ITEMS + [
        ExplainItem(_tx("TX-3", "300", "USD"), FX_PROPOSAL, "Acme"),
        ExplainItem(_tx("TX-4", "800"), PARTIAL_EXCEPTION, "בטא שיווק"),
    ]
    totals = compute_totals(items, filtered=3)
    assert (totals.total, totals.certain, totals.probable, totals.exceptions, totals.filtered) == (7, 0, 2, 2, 3)
    assert totals.proposed == (Money.of("7037.80", "ILS"), Money.of("300", "USD"))  # חריגים לא נכנסים לסכום


def test_extract_numbers_normalizes():
    assert extract_numbers("7,037.80 ₪, 61.60 (0.88%) INV-10026") == {"7037.8", "61.6", "0.88", "10026"}


# ---------- explain.py: הצומת ----------


def test_proposals_get_template_and_are_not_sent_to_model():
    model = FakeModel(_answer({"TX-2": "בטא שיווק שילמה 500 ₪ בלי לציין חשבונית; כדאי לברר על מה התשלום."}))
    result = explain_run(model, ITEMS, compute_totals(ITEMS, 3), POLICY)
    assert result.sentences["TX-1"] == template_sentence(ITEMS[0])
    assert "TX-1" not in model.messages[1][1]  # ההצעה לא נשלחה למודל
    assert "TX-2" in model.messages[1][1]


def test_exception_sentence_with_real_numbers_is_kept():
    sentence = "בטא שיווק שילמה 500 ₪ בלי לציין חשבונית; כדאי לברר על מה התשלום."
    result = explain_run(FakeModel(_answer({"TX-2": sentence})), ITEMS, compute_totals(ITEMS, 3), POLICY)
    assert result.sentences["TX-2"] == sentence
    assert result.replaced == ()


def test_invented_number_falls_back_to_template():
    model = FakeModel(_answer({"TX-2": "שולמו 510 ₪ בלי מספר חשבונית."}))  # הסכום האמיתי 500
    result = explain_run(model, ITEMS, compute_totals(ITEMS, 3), POLICY)
    assert result.sentences["TX-2"] == template_sentence(ITEMS[1])
    assert result.replaced == ("TX-2",)


def test_missing_sentence_falls_back_to_template():
    result = explain_run(FakeModel(_answer({})), ITEMS, compute_totals(ITEMS, 3), POLICY)
    assert result.sentences["TX-2"] == template_sentence(ITEMS[1])
    assert "TX-2" in result.replaced


def test_policy_threshold_is_allowed():
    items = [ExplainItem(_tx("TX-5", "3246"), FX_EXCEPTION, "Acme")]
    sentence = "הפרש שער של 6.53% בחשבונית INV-10031, מעל הסף של 2%; כדאי לבדוק עם הלקוח."
    result = explain_run(FakeModel(_answer({"TX-5": sentence})), items, compute_totals(items, 0), POLICY)
    assert result.sentences["TX-5"] == sentence


def test_summary_with_invented_number_falls_back():
    totals = compute_totals(ITEMS, 3)
    model = FakeModel(_answer({"TX-2": "בטא שיווק שילמה 500 ₪ בלי חשבונית."}, summary="טופלו 9 תנועות."))
    result = explain_run(model, ITEMS, totals, POLICY)
    assert result.summary == template_summary(totals)
    assert SUMMARY_KEY in result.replaced


def test_no_items_does_not_call_model():
    model = FakeModel(_answer({}))
    result = explain_run(model, [], compute_totals([], 2), POLICY)
    assert model.calls == 0
    assert result.sentences == {}


def test_fallback_explanations_uses_only_templates():
    totals = compute_totals(ITEMS, 3)
    result = fallback_explanations(ITEMS, totals)
    assert result.model_failed
    assert result.sentences == {i.tx.tx_id: template_sentence(i) for i in ITEMS}
    assert result.summary == template_summary(totals)
