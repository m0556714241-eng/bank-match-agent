"""
צומת הסינון (סעיף 4, ה-52): האם התנועה היא תקבול מלקוח.
"""

from __future__ import annotations

from src.domain.models import BankTransaction
from src.engine.policy import MatchPolicy
from src.engine.results import FilterOutcome, FilterReason, FilterResult


def filter_tx(tx: BankTransaction, policy: MatchPolicy) -> FilterResult:
    """מסווג תנועה: תקבול (ממשיכים), לא תקבול (מסוננת), או סוג שלא נתמך (חריג)."""
    text = tx.bank_text

    # סדר הבדיקות קבוע: קודם הכי ודאי (סימן הסכום), אחר כך לפי הטקסט
    if not tx.is_incoming:
        return FilterResult(FilterOutcome.FILTERED, FilterReason.OUTGOING)
    if _contains_any(text, policy.internal_transfer_keywords):
        return FilterResult(FilterOutcome.FILTERED, FilterReason.INTERNAL_TRANSFER)
    if _contains_any(text, policy.interest_keywords):
        return FilterResult(FilterOutcome.FILTERED, FilterReason.INTEREST)
    if _contains_any(text, policy.tax_refund_keywords):
        return FilterResult(FilterOutcome.FILTERED, FilterReason.TAX_REFUND)

    # זיכוי סליקה הוא כסף אמיתי שנכנס - לא מסננים בשקט, אלא מעבירים לאדם
    if tx.account_id == policy.card_settlement_account and policy.card_settlement_keyword in text:
        return FilterResult(FilterOutcome.UNSUPPORTED)

    return FilterResult(FilterOutcome.PASS)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """עזר של filter_tx: האם אחת ממילות המפתח מופיעה בטקסט."""
    return any(keyword in text for keyword in keywords)
