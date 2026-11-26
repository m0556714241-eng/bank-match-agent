"""
הצומת explain_and_summarize (שלב 5, ה-08, ה-67): רץ פעם אחת בכל ריצה,
אחרי שכל התנועות טופלו ולפני העצירה לאישור אדם.

שלושה שלבים:
1. הקוד כותב משפט מתבנית לכל שורה, וסיכום מתבנית - תמיד, לפני המודל.
   הצעות (ודאיות וסבירות) מקבלות את משפט התבנית כמו שהוא - בלי מודל (ה-69).
2. המודל כותב הסבר רק לחריגים, וסיכום ריצה - בקריאה אחת.
3. הקוד בודק כל מספר שהמודל כתב. מספר שלא קיים בנתונים -> המשפט הזה חוזר לתבנית.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field

from src.adapters.interfaces import AdjustmentType, Certainty
from src.domain.frozen import freeze_sequences
from src.domain.models import BankTransaction
from src.domain.money import Money
from src.engine.policy import MatchPolicy
from src.engine.results import ExceptionReason, InvoiceMatch, MatchKind
from src.llm.customer_llm import StructuredChatModel

MAX_TOKENS = 4000  # משפטי החריגים + הסיכום בתשובה אחת (ה-66)
SUMMARY_KEY = "SUMMARY"  # המזהה של הסיכום ברשימת המשפטים שהוחלפו

# ---------- התבניות (שלב 1) ----------

# הצעה (ודאית / סבירה) - לפי הכלל שמצא אותה
KIND_TEMPLATES: dict[MatchKind, str] = {
    MatchKind.EXACT_REF: "התאמה מדויקת לחשבונית {invoices} שצוינה בטקסט, בסכום {open_total}.",
    MatchKind.EXACT: "התאמה מדויקת לחשבונית {invoices} בסכום {open_total}.",
    MatchKind.COMBINATION: "התאמה מדויקת לצירוף החשבוניות {invoices}, ביחד {open_total}.",
    MatchKind.BANK_FEE: "התאמה לחשבונית {invoices} ({open_total}); החסר {fee} נרשם כעמלת בנק.",
    MatchKind.WITHHOLDING: "התאמה לחשבונית {invoices} ({open_total}); החסר {withholding} נרשם כניכוי במקור.",
    MatchKind.FX: "חשבונית {invoice} ({invoice_amount}) שולמה בשקלים; הפרש שער {fx_diff} ({fx_diff_pct}).",
    MatchKind.PARTIAL_REF: "תשלום חלקי לחשבונית {invoice}; נשאר פתוח {remaining}.",
}

# חריג - לפי הסיבה (סעיף 9)
EXCEPTION_TEMPLATES: dict[ExceptionReason, str] = {
    ExceptionReason.CUSTOMER_NOT_IDENTIFIED: "לא זוהה לקוח לפי טקסט הבנק; יש לברר מי שילם {amount}.",
    ExceptionReason.LLM_SUGGESTION_NOT_VERIFIED: "המודל הציע את {customer}, אך אין לו חשבונית פתוחה שמתאימה ל-{amount}; יש לבדוק.",
    ExceptionReason.DIFF_NOT_EXPLAINED: "הסכום {amount} לא תואם לחשבוניות הפתוחות של {customer}, וההפרש לא מוסבר.",
    ExceptionReason.AMBIGUOUS_MATCH: "הסכום {amount} מתאים ליותר מלקוח אחד ({candidates}); יש להחליט ידנית.",
    ExceptionReason.OVERPAYMENT: "הסכום {amount} גבוה מהחשבוניות הפתוחות של {customer}{duplicate}.",
    ExceptionReason.PARTIAL_WITHOUT_REFERENCE: (
        "סכום חלקי ({amount}) שלא תואם לאף חשבונית של {customer}, ולא כתוב על איזו חשבונית - "
        "אי אפשר לדעת על מה להחיל."
    ),
    ExceptionReason.FX_DIFF_ABOVE_TOLERANCE: (
        "חשבונית {invoice} שולמה בשקלים עם הפרש שער {fx_diff} ({fx_diff_pct}), מעל הסף; יש לבדוק עם הלקוח."
    ),
    ExceptionReason.UNSUPPORTED_TX_TYPE: "סוג תנועה שלא נתמך (למשל סליקת אשראי); יש לטפל ידנית ב-{amount}.",
    ExceptionReason.FX_RATE_UNAVAILABLE: "אין שער יציג לתאריך הנדרש, ולכן אי אפשר לחשב התאמה; יש לבדוק ידנית.",
    ExceptionReason.INVOICE_CONFLICT: (
        "החשבונית {invoice} כבר הוצעה להעברה {other_tx} באותה ריצה; ייתכן תשלום כפול - יש לבדוק."
    ),
}

SUMMARY_TEMPLATE = (
    "טופלו {total} תנועות: {certain} ודאיות, {probable} סבירות ו-{exceptions} חריגים; "
    "{filtered} סוננו כי אינן תקבולים. סכום ההצעות: {proposed}."
)

# ---------- הוראות למודל (שלב 2) ----------

SYSTEM_PROMPT = """You explain exceptions to an accountant who reviews bank reconciliation results.
Every line you get is an exception: an incoming payment the system could not match safely.
For each line you get: its id, the customer, a template sentence and the facts behind it.
Rules:
- Write each sentence in clear, natural Hebrew. One sentence, at most 25 words.
- Keep the meaning of the template. Say what the problem is and what the accountant should check.
- Use ONLY numbers and invoice numbers that appear in that line's template or facts. Never calculate new numbers.
- Do not recommend amounts, postings or decisions about money.
- Return exactly one sentence for every line id you received.
- Also write a run summary in Hebrew, 2-3 sentences, using ONLY the numbers in the run totals.
  Point out what needs attention today, for example several exceptions of the same customer."""


# ---------- מבנים ----------


@dataclass(frozen=True)
class ExplainItem:
    """שורה אחת להסבר: התנועה, מה המנוע החליט עליה, ושם הלקוח לתצוגה."""

    tx: BankTransaction  # התנועה מהבנק
    match: InvoiceMatch  # ההצעה או החריג מהמנוע
    customer_name: str | None = None  # שם הלקוח (None = לא זוהה)


@dataclass(frozen=True)
class RunTotals:
    """המספרים של סיכום הריצה - הקוד מחשב, המודל רק מנסח."""

    total: int  # כמה תנועות נכנסו לריצה
    certain: int  # ודאיות
    probable: int  # סבירות
    exceptions: int  # חריגים
    filtered: int  # סוננו (לא תקבולים)
    proposed: tuple[Money, ...] = ()  # סכום ההצעות (ודאיות + סבירות), אחד לכל מטבע

    def __post_init__(self) -> None:
        freeze_sequences(self)  # tuple נשאר tuple גם אחרי טעינה מקובץ השמירה (ה-71)


@dataclass(frozen=True)
class ExplanationResult:
    """מה הצומת מחזיר לגרף - מה שמסך האישור יציג."""

    sentences: dict[str, str]  # מזהה תנועה -> המשפט הסופי (הצעה: תבנית; חריג: המודל או התבנית)
    summary: str  # סיכום הריצה הסופי
    replaced: tuple[str, ...] = ()  # שורות שבהן משפט המודל נפסל וחזר לתבנית (נשמר ב-LangSmith)
    model_failed: bool = False  # True = הגיבוי הופעל, הכל מתבניות

    def __post_init__(self) -> None:
        freeze_sequences(self)  # tuple נשאר tuple גם אחרי טעינה מקובץ השמירה (ה-71)


class LineExplanation(BaseModel):
    """משפט אחד שהמודל מחזיר."""

    tx_id: str = Field(description="the line id exactly as received")
    sentence: str = Field(description="one Hebrew sentence, at most 25 words")


class ExplanationBatch(BaseModel):
    """כל התשובה של המודל: משפט לכל שורה + סיכום."""

    lines: list[LineExplanation]
    summary: str = Field(description="run summary in Hebrew, 2-3 sentences")


# ---------- שלב 1: תבניות ----------


def template_sentence(item: ExplainItem) -> str:
    """המשפט מהתבנית לשורה אחת - תמיד נכון, כי כל המספרים בו מגיעים מהקוד."""
    m = item.match
    template = EXCEPTION_TEMPLATES[m.exception_reason] if m.exception_reason else KIND_TEMPLATES[m.kind]
    sentence = template.format_map(_template_fields(item))
    return " ".join([sentence, *m.notes])  # הערות המנוע (למשל "היו עוד אפשרויות") נכנסות כמו שהן


def compute_totals(items: list[ExplainItem], filtered: int) -> RunTotals:
    """סופר וסוכם את הריצה. סכום ההצעות - לפי מטבע, כי אסור לחבר שקלים לדולרים."""
    counts = Counter(i.match.certainty for i in items)
    sums: dict[str, Money] = {}
    for i in items:
        if not i.match.is_exception:
            money = i.tx.money
            sums[money.currency] = sums[money.currency] + money if money.currency in sums else money
    return RunTotals(
        total=len(items) + filtered,
        certain=counts[Certainty.CERTAIN],
        probable=counts[Certainty.PROBABLE],
        exceptions=counts[Certainty.EXCEPTION],
        filtered=filtered,
        proposed=tuple(sums[c] for c in sorted(sums)),
    )


def template_summary(totals: RunTotals) -> str:
    """הסיכום מהתבנית."""
    proposed = ", ".join(str(m) for m in totals.proposed) or "0"
    return SUMMARY_TEMPLATE.format(
        total=totals.total,
        certain=totals.certain,
        probable=totals.probable,
        exceptions=totals.exceptions,
        filtered=totals.filtered,
        proposed=proposed,
    )


# ---------- שלב 3: בדיקת מספרים ----------


def extract_numbers(text: str) -> set[str]:
    """כל המספרים בטקסט, בצורה אחידה: 7,037.80 -> 7037.8 ; 0.88% -> 0.88 ; INV-10026 -> 10026."""
    found = set()
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            found.add(format(Decimal(token.replace(",", "")).normalize(), "f"))
        except InvalidOperation:
            continue
    return found


def allowed_numbers(item: ExplainItem, policy: MatchPolicy) -> set[str]:
    """המספרים שמותר למשפט של השורה הזו להכיל: כל מה שיש בתבנית, בעובדות, בהחלות ובהפרשים."""
    m = item.match
    sources = [template_sentence(item), str(item.tx.money), *m.evidence.values(), *m.notes]
    sources += [f"{a.invoice_id} {a.amount}" for a in m.applications]
    sources += [str(a.amount) for a in m.adjustments]
    if m.receipt_amount is not None:
        sources.append(str(m.receipt_amount))
    sources += _policy_thresholds(item, policy)  # מותר להזכיר את הסף ("מעל 2%")
    return set().union(*(extract_numbers(s) for s in sources))


def unsupported_numbers(sentence: str, allowed: set[str]) -> set[str]:
    """המספרים במשפט שלא מופיעים בנתונים. ריק = המשפט עבר."""
    return extract_numbers(sentence) - allowed


# ---------- הצומת ----------


def build_messages(exceptions: list[ExplainItem], totals: RunTotals) -> list[tuple[str, str]]:
    """ההודעות למודל: הוראות קבועות + לכל חריג התבנית והעובדות + מספרי הסיכום."""
    blocks = []
    for item in exceptions:
        m = item.match
        facts = "; ".join(f"{k}={v}" for k, v in m.evidence.items()) or "-"
        blocks.append(
            f"[{item.tx.tx_id}] reason={m.exception_reason} | customer={item.customer_name or '-'} | "
            f"bank_amount={item.tx.money}\n  template: {template_sentence(item)}\n  facts: {facts}"
        )
    lines = "\n".join(blocks) or "(no exceptions today)"
    human = "Exceptions:\n" + lines + f"\n\nRun totals (template): {template_summary(totals)}"
    return [("system", SYSTEM_PROMPT), ("human", human)]


def explain_run(
    llm: StructuredChatModel,
    items: list[ExplainItem],
    totals: RunTotals,
    policy: MatchPolicy,
) -> ExplanationResult:
    """הלב של הצומת: הצעות -> תבנית; חריגים וסיכום -> קריאה אחת למודל -> בדיקת כל מספר."""
    if not items:  # אין שורות להסביר - לא קוראים למודל
        return ExplanationResult({}, template_summary(totals))

    # הצעות: משפט התבנית כמו שהוא - בלי מודל (ה-69)
    sentences = {i.tx.tx_id: template_sentence(i) for i in items if not i.match.is_exception}
    exceptions = [i for i in items if i.match.is_exception]

    answer = llm.with_structured_output(ExplanationBatch).invoke(build_messages(exceptions, totals))
    from_model = {line.tx_id: line.sentence.strip() for line in answer.lines}

    replaced: list[str] = []
    for item in exceptions:
        tx_id = item.tx.tx_id
        candidate = from_model.get(tx_id, "")
        if candidate and not unsupported_numbers(candidate, allowed_numbers(item, policy)):
            sentences[tx_id] = candidate  # המשפט של המודל עבר את הבדיקה
        else:
            sentences[tx_id] = template_sentence(item)  # חסר או עם מספר שלא קיים -> תבנית
            replaced.append(tx_id)

    summary = answer.summary.strip()
    if not summary or unsupported_numbers(summary, extract_numbers(template_summary(totals))):
        summary = template_summary(totals)
        replaced.append(SUMMARY_KEY)

    return ExplanationResult(sentences, summary, tuple(replaced))


def fallback_explanations(items: list[ExplainItem], totals: RunTotals) -> ExplanationResult:
    """הגיבוי הבטוח (ה-66): המודל לא ענה אחרי כל הניסיונות -> הכל מתבניות."""
    sentences = {i.tx.tx_id: template_sentence(i) for i in items}
    return ExplanationResult(sentences, template_summary(totals), model_failed=True)


# ---------- עזר ----------


class _Fields(dict):
    """עזר של template_sentence: שדה שחסר בעובדות מופיע כ"-" במקום לזרוק שגיאה."""

    def __missing__(self, key: str) -> str:
        return "-"


def _template_fields(item: ExplainItem) -> _Fields:
    """עזר של template_sentence: כל הערכים שהתבניות יכולות להשתמש בהם."""
    m = item.match
    fields = _Fields(m.evidence)
    fields["amount"] = str(item.tx.money)
    fields["customer"] = item.customer_name or "הלקוח"
    for adj in m.adjustments:
        if adj.type == AdjustmentType.BANK_FEE:
            fields["fee"] = str(adj.amount)
        elif adj.type == AdjustmentType.WITHHOLDING_TAX:
            fields["withholding"] = str(adj.amount)
    dup = m.evidence.get("possible_duplicate_of")
    fields["duplicate"] = f"; ייתכן תשלום כפול של {dup}" if dup else ""
    return fields


def _policy_thresholds(item: ExplainItem, policy: MatchPolicy) -> list[str]:
    """עזר של allowed_numbers: הספים מהמדיניות שרלוונטיים לשורה (אחוזים ותקרת עמלה)."""
    thresholds = [f"{policy.fx_diff_tolerance * 100}", f"{policy.fx_match_tolerance * 100}"]
    fee_limit = policy.fee_limits.get(item.tx.money.currency)
    if fee_limit is not None:
        thresholds.append(str(fee_limit))
    return thresholds
