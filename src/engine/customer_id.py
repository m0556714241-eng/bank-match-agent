"""
צמתי זיהוי הלקוח (סעיף 5, ה-53):
- identify_customer: זיהוי בקוד, לפי סדר קבוע.
- verify_llm_customer: אימות בקוד של הלקוח שמודל השפה הציע (שלב 5).
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from src.domain.models import Alias, BankTransaction, Customer, OpenInvoice
from src.engine.policy import MatchPolicy
from src.engine.results import Candidate, CustomerMatch, CustomerOutcome, IdentifiedBy

# סיומות משפטיות שלא עוזרות לזהות (אחרי הסרת גרשיים: בע״מ -> בעמ)
_LEGAL_SUFFIXES = {"בעמ", "ltd", "inc", "llc", "gmbh", "ab", "co", "corp"}

# גרשיים, גרש ומירכאות בכל הצורות (עברית ולועזית) - נמחקים
_QUOTES = "\"'`״׳‘’“”"

# מקף ונקודה מחברים (שופ-נט -> שופנט, ש.נ. -> שנ); שאר הסימנים -> רווח
_JOINERS = "-–—."


def normalize_name(text: str, invoice_pattern: str = r"INV-\d{5}") -> str:
    """מנרמל שם להשוואה: בלי מספרי חשבונית, גרשיים, מקפים, סיומות משפטיות ואותיות גדולות."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(invoice_pattern, " ", text)  # מספר חשבונית הוא לא חלק מהשם
    text = text.translate(str.maketrans("", "", _QUOTES + _JOINERS))
    text = re.sub(r"[^\w\s]", " ", text)  # כל סימן אחר -> רווח
    words = [w for w in text.lower().split() if w not in _LEGAL_SUFFIXES]
    return " ".join(words)


def identify_customer(
    tx: BankTransaction,
    customers: list[Customer],
    aliases: list[Alias],
    open_invoices: list[OpenInvoice],
    policy: MatchPolicy,
) -> CustomerMatch:
    """זיהוי לקוח בקוד: מספר חשבונית -> שם זהה -> כינוי -> דמיון. לא בטוח -> צריך מודל."""
    refs = tuple(dict.fromkeys(re.findall(policy.invoice_number_pattern, tx.bank_text)))

    # 1. מספר חשבונית בטקסט - הדרך הכי ודאית
    by_ref = _identify_by_invoice_ref(refs, open_invoices)
    if by_ref:
        return CustomerMatch(CustomerOutcome.IDENTIFIED, by_ref, IdentifiedBy.INVOICE_REF, strong=True, invoice_refs=refs)

    text = normalize_name(tx.bank_text, policy.invoice_number_pattern)

    # 2. שם זהה לשם של לקוח
    exact = tuple(c.customer_id for c in customers if normalize_name(c.name) == text)
    if len(exact) == 1:
        return CustomerMatch(CustomerOutcome.IDENTIFIED, exact, IdentifiedBy.EXACT_NAME, strong=True, invoice_refs=refs)

    # 3. כינוי מוכר - יכול להצביע על כמה לקוחות (פייפלו), ואז הסכום יכריע
    matching_aliases = [a for a in aliases if normalize_name(a.bank_name) == text]
    if matching_aliases:
        ids = tuple(dict.fromkeys(a.customer_id for a in matching_aliases))
        confirmed = min(a.times_confirmed for a in matching_aliases)
        strong = confirmed >= policy.alias_confirmations_for_certain
        return CustomerMatch(CustomerOutcome.IDENTIFIED, ids, IdentifiedBy.ALIAS, strong=strong, invoice_refs=refs)

    # 4. דמיון שמות
    scored = _score_customers(text, customers, aliases)
    top = scored[0]
    second = scored[1].score if len(scored) > 1 else 0.0

    if top.score >= policy.fuzzy_accept and top.score - second >= policy.fuzzy_min_gap:
        return CustomerMatch(CustomerOutcome.IDENTIFIED, (top.customer_id,), IdentifiedBy.FUZZY, invoice_refs=refs)

    # לא בטוח -> מודל השפה. אילו מועמדים לשלוח?
    if top.score >= policy.fuzzy_accept:  # כמה לקוחות קרובים (בטא-טק / בטא שיווק)
        candidates = tuple(c for c in scored if top.score - c.score < policy.fuzzy_min_gap)
    elif top.score >= policy.fuzzy_candidates:
        candidates = tuple(c for c in scored if c.score >= policy.fuzzy_candidates)
    else:
        candidates = tuple(scored)  # אין שם דומה - כל הלקוחות
    return CustomerMatch(CustomerOutcome.NEEDS_LLM, candidates=candidates, invoice_refs=refs)


def verify_llm_customer(customer_id: str | None, customers: list[Customer], refs: tuple[str, ...] = ()) -> CustomerMatch:
    """אימות הצעת המודל: הלקוח קיים ברשימה? (ההתאמה לחשבונית נבדקת בצומת הבא)."""
    known = {c.customer_id for c in customers}
    if customer_id is None or customer_id not in known:
        return CustomerMatch(CustomerOutcome.NOT_VERIFIED, invoice_refs=refs)
    return CustomerMatch(CustomerOutcome.IDENTIFIED, (customer_id,), IdentifiedBy.LLM, invoice_refs=refs)


# ---------- עזר ----------


def _identify_by_invoice_ref(refs: tuple[str, ...], open_invoices: list[OpenInvoice]) -> tuple[str, ...]:
    """עזר של identify_customer: הלקוחות של החשבוניות הפתוחות שמספרן מופיע בטקסט."""
    by_id = {inv.invoice_id: inv.customer_id for inv in open_invoices}
    owners = tuple(dict.fromkeys(by_id[r] for r in refs if r in by_id))
    return owners if len(owners) == 1 else ()  # חשבוניות של כמה לקוחות - לא סומכים על המספרים


def _score_customers(text: str, customers: list[Customer], aliases: list[Alias]) -> list[Candidate]:
    """עזר של identify_customer: ציון דמיון לכל לקוח (הגבוה מבין השם והכינויים), מהגבוה לנמוך."""
    names: dict[str, list[str]] = {c.customer_id: [normalize_name(c.name)] for c in customers}
    for a in aliases:
        names.setdefault(a.customer_id, []).append(normalize_name(a.bank_name))
    scored = [
        Candidate(cid, round(max(fuzz.WRatio(text, n) for n in options), 1)) for cid, options in names.items()
    ]
    return sorted(scored, key=lambda c: (-c.score, c.customer_id))
