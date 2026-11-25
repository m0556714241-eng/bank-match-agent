"""
צומת מציאת החשבונית וסיווג ההפרש (סעיפים 6–8, ה-54 עד ה-57).

לכל לקוח שזוהה: בודקים את הכללים לפי סדר קבוע, והראשון שמתאים מנצח.
שום כלל לא מנחש: מקרה שהמדיניות לא מגדירה -> חריג.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations

from src.adapters.interfaces import (
    Adjustment,
    AdjustmentType,
    Application,
    Certainty,
    FxRateProvider,
    FxRateUnavailable,
)
from src.domain.models import BankTransaction, Customer, Invoice, OpenInvoice
from src.domain.money import Money, convert
from src.engine.policy import MatchPolicy
from src.engine.results import CustomerMatch, ExceptionReason, IdentifiedBy, InvoiceMatch, MatchKind


@dataclass(frozen=True)
class _Group:
    """עזר: קבוצה של חשבונית אחת או יותר, והיתרה הפתוחה הכוללת שלה."""

    invoices: tuple[OpenInvoice, ...]
    total: Money

    @property
    def ids(self) -> str:
        return ", ".join(i.invoice_id for i in self.invoices)

    @property
    def age_key(self) -> tuple:
        # לבחירת "הוותיקה": קודם התאריך הוותיק ביותר, אחר כך פחות חשבוניות
        return (tuple(sorted(i.issue_date for i in self.invoices)), len(self.invoices))


def find_invoice(
    tx: BankTransaction,
    match: CustomerMatch,
    open_invoices: list[OpenInvoice],
    customers: list[Customer],
    fx: FxRateProvider,
    policy: MatchPolicy,
    paid_invoices: list[Invoice] = (),
) -> InvoiceMatch:
    """מוצא חשבונית/צירוף ומסווג את ההפרש. כמה לקוחות מועמדים (פייפלו) -> הסכום מכריע."""
    by_customer = {c.customer_id: c for c in customers}
    results = {
        cid: _match_customer(tx, match, by_customer[cid], open_invoices, fx, policy, paid_invoices)
        for cid in match.customer_ids
    }
    if len(results) == 1:
        return next(iter(results.values()))

    # כינוי משותף: רק לקוח אחד שיש לו התאמה -> הוא. כמה -> חריג. אף אחד -> לא מוסבר
    found = {cid: r for cid, r in results.items() if not r.is_exception}
    if len(found) == 1:
        return next(iter(found.values()))
    if len(found) > 1:
        return _exception(
            ExceptionReason.AMBIGUOUS_MATCH,
            evidence={"candidates": ", ".join(found), "amount": str(tx.money)},
        )
    return _exception(ExceptionReason.DIFF_NOT_EXPLAINED, evidence={"candidates": ", ".join(results)})


def _match_customer(
    tx: BankTransaction,
    match: CustomerMatch,
    customer: Customer,
    all_open: list[OpenInvoice],
    fx: FxRateProvider,
    policy: MatchPolicy,
    paid_invoices: list[Invoice],
) -> InvoiceMatch:
    """הכללים של סעיף 6 ללקוח אחד, לפי הסדר."""
    cid = customer.customer_id
    mine = sorted((i for i in all_open if i.customer_id == cid), key=lambda i: (i.issue_date, i.invoice_id))
    same_ccy = [i for i in mine if i.currency == tx.currency]
    amount = tx.money

    # 1. מספר חשבונית בטקסט
    referenced = [i for i in same_ccy if i.invoice_id in match.invoice_refs]
    if referenced:
        return _match_referenced(tx, match, customer, referenced, policy)

    # 2. חשבונית אחת בסכום זהה  3. צירוף בסכום זהה
    exact = [g for g in _groups(same_ccy, policy.max_combination_size) if g.total == amount]
    if exact:
        best = _oldest(exact)
        kind = MatchKind.EXACT if len(best.invoices) == 1 else MatchKind.COMBINATION
        return _proposal(tx, match, cid, best, kind, alternatives=len(exact) - 1)

    # 4. עמלת בנק: חסר עד הסף של המטבע
    fee_limit = policy.fee_limits[tx.currency]
    fee = [g for g in _groups(same_ccy, policy.max_combination_size) if amount < g.total <= amount + fee_limit]
    if fee:
        best = min(fee, key=lambda g: (g.total - amount, g.age_key))
        adj = (Adjustment(AdjustmentType.BANK_FEE, best.total - amount),)
        return _proposal(tx, match, cid, best, MatchKind.BANK_FEE, adjustments=adj, alternatives=len(fee) - 1)

    # 5. ניכוי במקור: רק ללקוח עם שיעור ניכוי
    if customer.withholding_rate > 0:
        wh = _withholding_candidates(same_ccy, amount, customer.withholding_rate, policy)
        if wh:
            best = min(wh, key=lambda g: (abs(g.total.scaled(1 - customer.withholding_rate) - amount).amount, g.age_key))
            adjs = _withholding_adjustments(best, amount, customer.withholding_rate)
            return _proposal(tx, match, cid, best, MatchKind.WITHHOLDING, adjustments=adjs, alternatives=len(wh) - 1)

    # 6. מטבע חוץ ששולם בשקלים
    foreign = [i for i in mine if i.currency != tx.currency]
    if foreign:
        fx_result = _match_fx(tx, match, cid, foreign, fx, policy)
        if fx_result is not None:
            return fx_result

    # 7. אין התאמה - איזה חריג?
    return _no_match(tx, match, cid, same_ccy, paid_invoices)


# ---------- כלל 1: מספר חשבונית בטקסט ----------


def _match_referenced(
    tx: BankTransaction, match: CustomerMatch, customer: Customer, referenced: list[OpenInvoice], policy: MatchPolicy
) -> InvoiceMatch:
    """עזר של _match_customer: הסכום מול החשבוניות שהלקוח ציין - מלא, הפרש מוסבר, חלקי או יתר."""
    group = _Group(tuple(referenced), _sum(referenced, tx.money))
    amount, cid = tx.money, customer.customer_id

    if amount == group.total:
        return _proposal(tx, match, cid, group, MatchKind.EXACT_REF)
    if amount < group.total <= amount + policy.fee_limits[tx.currency]:
        adj = (Adjustment(AdjustmentType.BANK_FEE, group.total - amount),)
        return _proposal(tx, match, cid, group, MatchKind.BANK_FEE, adjustments=adj)
    if customer.withholding_rate > 0 and _withholding_candidates(list(referenced), amount, customer.withholding_rate, policy, only=group):
        adjs = _withholding_adjustments(group, amount, customer.withholding_rate)
        return _proposal(tx, match, cid, group, MatchKind.WITHHOLDING, adjustments=adjs)
    if amount > group.total:
        return _exception(ExceptionReason.OVERPAYMENT, cid, {"invoices": group.ids, "open_total": str(group.total)})
    if len(referenced) == 1:
        # תשלום חלקי עם מספר חשבונית: מחילים את מה ששולם, החשבונית נשארת פתוחה
        inv = referenced[0]
        app = (Application(inv.invoice_id, amount),)
        return InvoiceMatch(
            certainty=Certainty.PROBABLE,
            customer_id=cid,
            kind=MatchKind.PARTIAL_REF,
            receipt_amount=amount,
            applications=app,
            evidence={"invoice": inv.invoice_id, "open_before": str(inv.open_money), "remaining": str(inv.open_money - amount)},
        )
    return _exception(ExceptionReason.DIFF_NOT_EXPLAINED, cid, {"invoices": group.ids, "open_total": str(group.total)})


# ---------- כלל 6: מטבע חוץ ----------


def _match_fx(
    tx: BankTransaction, match: CustomerMatch, cid: str, foreign: list[OpenInvoice], fx: FxRateProvider, policy: MatchPolicy
) -> InvoiceMatch | None:
    """עזר של _match_customer: חשבונית במט"ח ששולמה בשקלים (סעיף 8, ה-56). None = אין התאמה."""
    try:
        candidates = []
        for inv in foreign:
            pay_rate = fx.rate(inv.currency, tx.tx_date)
            expected = convert(inv.open_money, pay_rate.rate_to_ils, tx.currency)
            deviation = abs((tx.money - expected).amount) / expected.amount
            if deviation <= policy.fx_match_tolerance:
                candidates.append((deviation, inv, pay_rate))
        if not candidates:
            return None
        _, inv, pay_rate = min(candidates, key=lambda c: (c[0], c[1].issue_date))
        invoice_rate = fx.rate(inv.currency, inv.issue_date)
    except FxRateUnavailable as exc:
        return _exception(ExceptionReason.FX_RATE_UNAVAILABLE, cid, {"error": str(exc)})

    booked = convert(inv.open_money, invoice_rate.rate_to_ils, tx.currency)  # ערך החשבונית ביום ההפקה
    diff = tx.money - booked  # חיובי = רווח שער, שלילי = הפסד שער
    pct = abs(diff.amount) / booked.amount
    evidence = {
        "invoice": inv.invoice_id,
        "invoice_amount": str(inv.open_money),
        "invoice_rate": f"{invoice_rate.rate_to_ils} ({invoice_rate.rate_date})",
        "payment_rate": f"{pay_rate.rate_to_ils} ({pay_rate.rate_date})",
        "booked_ils": str(booked),
        "fx_diff": str(diff),
        "fx_diff_pct": f"{pct * 100:.2f}%",
    }
    adjustments = (Adjustment(AdjustmentType.FX_DIFFERENCE, diff),) if not diff.is_zero() else ()
    within = pct <= policy.fx_diff_tolerance
    return InvoiceMatch(
        certainty=Certainty.PROBABLE if within else Certainty.EXCEPTION,
        customer_id=cid,
        kind=MatchKind.FX,
        receipt_amount=inv.open_money,
        applications=(Application(inv.invoice_id, inv.open_money),),
        adjustments=adjustments,
        exception_reason=None if within else ExceptionReason.FX_DIFF_ABOVE_TOLERANCE,
        evidence=evidence,
    )


# ---------- כלל 7: אין התאמה ----------


def _no_match(
    tx: BankTransaction, match: CustomerMatch, cid: str, same_ccy: list[OpenInvoice], paid_invoices: list[Invoice]
) -> InvoiceMatch:
    """עזר של _match_customer: בוחר את החריג המתאים כשאף כלל לא התאים (ה-57)."""
    amount = tx.money
    evidence = {"amount": str(amount), "open_invoices": ", ".join(f"{i.invoice_id}={i.open_money}" for i in same_ccy)}

    # תשלום כפול: אותו סכום כמו חשבונית של הלקוח ששולמה כבר
    duplicate = [p for p in paid_invoices if p.customer_id == cid and p.money == amount]
    if duplicate:
        evidence["possible_duplicate_of"] = duplicate[0].invoice_id
        return _exception(ExceptionReason.OVERPAYMENT, cid, evidence)

    # המודל הציע לקוח, אבל אין לו חשבונית מתאימה -> ההצעה לא אומתה (סעיף 5)
    if match.identified_by == IdentifiedBy.LLM:
        return _exception(ExceptionReason.LLM_SUGGESTION_NOT_VERIFIED, cid, evidence)

    total_open = _sum(same_ccy, amount)
    if amount > total_open:
        return _exception(ExceptionReason.OVERPAYMENT, cid, evidence)
    if any(amount < i.open_money for i in same_ccy):
        return _exception(ExceptionReason.PARTIAL_WITHOUT_REFERENCE, cid, evidence)
    return _exception(ExceptionReason.DIFF_NOT_EXPLAINED, cid, evidence)


# ---------- עזר כללי ----------


def _groups(invoices: list[OpenInvoice], max_size: int) -> Iterator[_Group]:
    """עזר: כל הצירופים של 1 עד max_size חשבוניות (חשבונית אחת היא "צירוף" בגודל 1)."""
    for size in range(1, min(max_size, len(invoices)) + 1):
        for combo in combinations(invoices, size):
            yield _Group(combo, _sum(combo, invoices[0].money))


def _withholding_candidates(
    invoices: list[OpenInvoice], amount: Money, rate: Decimal, policy: MatchPolicy, only: _Group | None = None
) -> list[_Group]:
    """עזר: קבוצות שהסכום שלהן כפול (1 - שיעור הניכוי) שווה לתשלום, בסטייה של עד 1."""
    groups = [only] if only is not None else list(_groups(invoices, policy.max_combination_size))
    return [g for g in groups if abs((g.total.scaled(1 - rate) - amount).amount) <= policy.withholding_tolerance]


def _withholding_adjustments(group: _Group, amount: Money, rate: Decimal) -> tuple[Adjustment, ...]:
    """עזר: ההפרשים של ניכוי במקור - הניכוי עצמו, ועיגול אם נשארה סטייה קטנה."""
    withheld = group.total.scaled(rate)  # למשל 10,000 × 3% = 300
    rounding = group.total - withheld - amount  # מה שנשאר אחרי הניכוי (עד 1)
    adjs = [Adjustment(AdjustmentType.WITHHOLDING_TAX, withheld)]
    if not rounding.is_zero():
        adjs.append(Adjustment(AdjustmentType.ROUNDING, rounding))
    return tuple(adjs)


def _oldest(groups: list[_Group]) -> _Group:
    """עזר: כשכמה קבוצות מתאימות לאותו לקוח - הוותיקה (ה-54, כמו NetSuite)."""
    return min(groups, key=lambda g: g.age_key)


def _sum(invoices, like: Money) -> Money:
    """עזר: סכום היתרות הפתוחות, במטבע של like."""
    total = Money.zero(like.currency)
    for inv in invoices:
        total = total + inv.open_money
    return total


def _proposal(
    tx: BankTransaction,
    match: CustomerMatch,
    cid: str,
    group: _Group,
    kind: MatchKind,
    adjustments: tuple[Adjustment, ...] = (),
    alternatives: int = 0,
) -> InvoiceMatch:
    """עזר: בונה הצעה. ודאית רק אם הזיהוי חזק, הסכום זהה, ולא הייתה אפשרות אחרת (סעיף 9)."""
    exact_kinds = (MatchKind.EXACT_REF, MatchKind.EXACT, MatchKind.COMBINATION)
    certain = match.strong and kind in exact_kinds and alternatives == 0
    notes = (f"היו עוד {alternatives} אפשרויות מתאימות - הוצעה הוותיקה",) if alternatives else ()
    return InvoiceMatch(
        certainty=Certainty.CERTAIN if certain else Certainty.PROBABLE,
        customer_id=cid,
        kind=kind,
        receipt_amount=group.total,
        applications=tuple(Application(i.invoice_id, i.open_money) for i in group.invoices),
        adjustments=adjustments,
        notes=notes,
        evidence={"invoices": group.ids, "open_total": str(group.total), "bank_amount": str(tx.money)},
    )


def _exception(reason: ExceptionReason, cid: str | None = None, evidence: dict[str, str] | None = None) -> InvoiceMatch:
    """עזר: בונה חריג עם הסיבה והעובדות."""
    return InvoiceMatch(certainty=Certainty.EXCEPTION, customer_id=cid, exception_reason=reason, evidence=evidence or {})
