"""
הצומת llm_identify (שלב 5, ה-65): כשהקוד לא בטוח מי הלקוח,
המודל מציע לקוח מתוך רשימה סגורה של מועמדים - או אומר "לא יודע".

המודל רק מציע. אחריו:
verify_llm_customer (הלקוח קיים?) -> find_invoice (יש לו חשבונית מתאימה?).
אם אין התאמה - חריג LLM_SUGGESTION_NOT_VERIFIED, ואדם מחליט (ה-06).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, create_model
from rapidfuzz import fuzz

from src.domain.models import Alias, BankTransaction, Customer
from src.engine.customer_id import normalize_name
from src.engine.policy import MatchPolicy
from src.engine.results import Candidate

UNKNOWN = "UNKNOWN"  # התשובה הלגיטימית של המודל כשהוא לא יודע
MAX_TOKENS = 300  # תשובה קצרה: מזהה + משפט אחד (ה-66)
MODEL_UNAVAILABLE_REASON = "המודל לא היה זמין"  # הערה לאדם כשכל הניסיונות נכשלו
NO_EVIDENCE_REASON = "המודל לא הביא ראיה תקפה מטקסט הבנק"  # הערה לאדם כשהסימוכין נפסלו (ה-73)

SYSTEM_PROMPT = """You identify which customer sent an incoming bank transfer.
You get the free text the bank attached to the transfer, and a closed list of candidate customers
(official name and names already seen in bank texts).
Rules:
- Answer ONLY with a customer_id from the list, or UNKNOWN.
- Choose a customer only if the bank text clearly refers to that customer.
- If two candidates fit equally, or none fits clearly, answer UNKNOWN. A wrong answer is worse than UNKNOWN.
- Ignore invoice numbers, account numbers and reference codes when comparing names.
- Evidence: in "quote" copy, word for word, the exact part of the bank text that refers to the customer,
  and in "matched_name" copy the exact official name or seen name from the list that it matches.
  If you answer UNKNOWN, leave both empty.
- Write the reason in Hebrew, one short sentence."""


class StructuredChatModel(Protocol):
    """מה שהצומת צריך ממודל: היכולת להחזיר פלט מובנה. ChatAnthropic עונה על זה, וגם המודל המדומה בבדיקות."""

    def with_structured_output(self, schema: type[BaseModel]) -> Any: ...


@dataclass(frozen=True)
class LlmCustomerSuggestion:
    """מה הצומת מחזיר לגרף."""

    customer_id: str | None  # הלקוח שהמודל הציע. None = לא יודע, או שהמודל נכשל, או שהסימוכין נפסלו
    reason: str  # הנימוק של המודל (או הערת הגיבוי) - יוצג לאדם ויישמר במעקב
    model_failed: bool = False  # True = הגיבוי הופעל (המודל לא ענה אחרי כל הניסיונות)
    quote: str = ""  # הסימוכין: הקטע מטקסט הבנק שהמודל הצביע עליו (ה-73) - יוצג במסך
    matched_name: str = ""  # השם מהרשימה שהציטוט מתאים לו (שם רשמי או כינוי)
    evidence_rejected: bool = False  # True = המודל בחר לקוח, אבל הסימוכין לא עברו את הבדיקה


def build_answer_schema(candidate_ids: tuple[str, ...]) -> type[BaseModel]:
    """בונה את מבנה התשובה לתנועה הזו: customer_id מותר רק מתוך המועמדים או UNKNOWN."""
    allowed = Literal[(*candidate_ids, UNKNOWN)]  # הרשימה הסגורה - נבנית מחדש לכל תנועה
    return create_model(
        "CustomerAnswer",
        customer_id=(allowed, Field(description="customer_id from the candidate list, or UNKNOWN")),
        reason=(str, Field(description="one short sentence in Hebrew explaining the choice")),
        quote=(str, Field(description="exact words copied from the bank text that refer to the customer, or empty")),
        matched_name=(str, Field(description="exact name from the candidate list that the quote matches, or empty")),
    )


def build_messages(
    tx: BankTransaction,
    candidates: tuple[Candidate, ...],
    customers: list[Customer],
    aliases: list[Alias],
) -> list[tuple[str, str]]:
    """בונה את ההודעות למודל: הוראות קבועות + טקסט התנועה ורשימת המועמדים."""
    names = {c.customer_id: c.name for c in customers}
    known: dict[str, list[str]] = {}
    for a in aliases:
        known.setdefault(a.customer_id, []).append(a.bank_name)

    lines = []
    for cand in candidates:
        seen = ", ".join(known.get(cand.customer_id, [])) or "-"
        lines.append(f"- {cand.customer_id} | official name: {names[cand.customer_id]} | seen in bank texts: {seen}")

    bank_text = tx.bank_text.strip() or "(empty)"
    human = f"Bank text: {bank_text}\n\nCandidates:\n" + "\n".join(lines)
    return [("system", SYSTEM_PROMPT), ("human", human)]


def suggest_customer(
    llm: StructuredChatModel,
    tx: BankTransaction,
    candidates: tuple[Candidate, ...],
    customers: list[Customer],
    aliases: list[Alias],
    policy: MatchPolicy,
) -> LlmCustomerSuggestion:
    """הלב של הצומת: קריאה אחת למודל עם פלט מובנה, ותרגום התשובה לתוצאה."""
    if not candidates:  # אין מועמדים - אין מה לשאול (וגם Literal ריק לא אפשרי)
        return LlmCustomerSuggestion(None, "אין מועמדים לבחירה")

    ids = tuple(c.customer_id for c in candidates)
    schema = build_answer_schema(ids)
    answer = llm.with_structured_output(schema).invoke(build_messages(tx, candidates, customers, aliases))

    if answer.customer_id == UNKNOWN:
        return LlmCustomerSuggestion(None, answer.reason.strip())

    quote, matched = answer.quote.strip(), answer.matched_name.strip()
    if not evidence_is_valid(tx.bank_text, quote, matched, answer.customer_id, customers, aliases, policy):
        # המודל בחר לקוח בלי הוכחה תקפה -> "לא יודע" -> חריג לאדם (ה-73)
        return LlmCustomerSuggestion(None, NO_EVIDENCE_REASON, quote=quote, matched_name=matched, evidence_rejected=True)
    return LlmCustomerSuggestion(answer.customer_id, answer.reason.strip(), quote=quote, matched_name=matched)


def evidence_is_valid(
    bank_text: str,
    quote: str,
    matched_name: str,
    customer_id: str,
    customers: list[Customer],
    aliases: list[Alias],
    policy: MatchPolicy,
) -> bool:
    """בדיקת הסימוכין: הציטוט בטקסט הבנק, השם שייך ללקוח שנבחר, והציטוט דומה לשם."""
    norm_quote, norm_name = normalize_name(quote), normalize_name(matched_name)
    if not norm_quote or not norm_name:
        return False  # בחר לקוח בלי להביא ציטוט ושם
    if norm_quote not in normalize_name(bank_text):
        return False  # הציטוט לא מופיע בטקסט הבנק - הומצא
    names = [c.name for c in customers if c.customer_id == customer_id]
    names += [a.bank_name for a in aliases if a.customer_id == customer_id]
    if norm_name not in {normalize_name(n) for n in names}:
        return False  # השם לא שייך ללקוח שנבחר
    return fuzz.partial_ratio(norm_quote, norm_name) >= policy.llm_quote_min_score  # הציטוט באמת דומה לשם


def fallback_suggestion() -> LlmCustomerSuggestion:
    """הגיבוי הבטוח (ה-66): כל הניסיונות נכשלו -> לא יודע -> התנועה תהיה חריג לאדם."""
    return LlmCustomerSuggestion(None, MODEL_UNAVAILABLE_REASON, model_failed=True)
