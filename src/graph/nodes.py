"""
הצמתים של הגרף (ה-49, ה-74): עטיפות קצרות סביב הפונקציות שכבר כתבנו.

כל עטיפה: לוקחת מהזיכרון (state) ומארגז הכלים (runtime.context) את מה שהפונקציה צריכה,
קוראת לה, ומחזירה עדכון לשדה אחד בזיכרון. הפונקציות עצמן נשארות בקבצים שלהן.

חלק 1 – צמתי התת-גרף (העברה אחת).
"""

from __future__ import annotations

from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langgraph.types import Command

from src.adapters.interfaces import Certainty
from src.engine.customer_id import identify_customer, verify_llm_customer
from src.engine.filters import filter_tx
from src.engine.invoice_match import find_invoice
from src.engine.results import ExceptionReason, FilterOutcome, InvoiceMatch
from src.graph.state import TxResult, TxState
from src.graph.toolbox import Toolbox
from src.llm.customer_llm import fallback_suggestion, suggest_customer

# ---------- צמתי התת-גרף: העברה אחת ----------


def filter_node(state: TxState, runtime: Runtime[Toolbox]) -> dict:
    """צומת הסינון: תקבול / לא תקבול / לא נתמך."""
    return {"filter": filter_tx(state["tx"], runtime.context.policy)}  # ההעברה + חוברת הנהלים -> תוצאת הסינון


def identify_node(state: TxState, runtime: Runtime[Toolbox]) -> dict:
    """צומת זיהוי הלקוח בקוד: חשבונית ← שם ← כינוי ← דמיון."""
    tools = runtime.context  # ארגז הכלים
    tx = state["tx"]  # ההעברה מהבנק
    match = identify_customer(
        tx,  # ההעברה
        tools.ledger.customers(),  # כל הלקוחות, מהמסד
        tools.ledger.aliases(),  # הכינויים שנלמדו, מהמסד
        tools.ledger.open_invoices(as_of=tx.tx_date),  # החשבוניות הפתוחות ביום ההעברה, מהמסד
        tools.policy,  # חוברת הנהלים (ספי הדמיון)
    )
    return {"customer": match}  # תוצאת הזיהוי -> לשדה customer


def llm_identify_node(state: TxState, runtime: Runtime[Toolbox]) -> dict:
    """צומת המודל: מציע לקוח מתוך המועמדים, עם סימוכין (ה-65, ה-73)."""
    tools = runtime.context  # ארגז הכלים
    suggestion = suggest_customer(
        tools.identify_model,  # המודל לזיהוי לקוח (Claude, 300 טוקנים)
        state["tx"],  # ההעברה מהבנק
        state["customer"].candidates,  # המועמדים שהזיהוי בקוד מצא
        tools.ledger.customers(),  # כל הלקוחות, מהמסד (לשמות הרשמיים)
        tools.ledger.aliases(),  # הכינויים, מהמסד (רמזים למודל + בדיקת הסימוכין)
        tools.policy,  # חוברת הנהלים (סף הסימוכין 70)
    )
    return {"llm": suggestion}  # הצעת המודל -> לשדה llm


def llm_identify_failed(state: TxState, error: NodeError) -> Command:
    """תוכנית החירום של צומת המודל: כל הניסיונות נכשלו ← "לא יודע" ← ממשיכים לאימות (ה-66)."""
    return Command(
        update={"llm": fallback_suggestion()},  # "לא יודע - המודל לא היה זמין" -> לשדה llm
        goto="verify_llm_customer",  # ממשיכים לאימות (שיחזיר "לא אומת" -> חריג)
    )


def verify_node(state: TxState, runtime: Runtime[Toolbox]) -> dict:
    """צומת האימות: הלקוח שהמודל הציע קיים? (ההתאמה לחשבונית נבדקת בצומת הבא)."""
    verified = verify_llm_customer(
        state["llm"].customer_id,  # הלקוח שהמודל הציע (None = לא יודע)
        runtime.context.ledger.customers(),  # כל הלקוחות, מהמסד
        state["customer"].invoice_refs,  # מספרי חשבונית שנמצאו בטקסט הבנק
    )
    return {"customer": verified}  # מחליף את תוצאת הזיהוי: "זוהה" או "לא אומת"


def find_invoice_node(state: TxState, runtime: Runtime[Toolbox]) -> dict:
    """צומת מציאת החשבונית וסיווג ההפרש."""
    tools = runtime.context  # ארגז הכלים
    tx = state["tx"]  # ההעברה מהבנק
    match = find_invoice(
        tx,  # ההעברה
        state["customer"],  # מי הלקוח (תוצאת הזיהוי / האימות)
        tools.ledger.open_invoices(as_of=tx.tx_date),  # החשבוניות הפתוחות ביום ההעברה, מהמסד
        tools.ledger.customers(),  # כל הלקוחות, מהמסד (לשיעור ניכוי במקור)
        tools.fx,  # שערי בנק ישראל (לחשבוניות במט"ח)
        tools.policy,  # חוברת הנהלים (עמלה, ניכוי, שער)
        tools.ledger.paid_invoices(as_of=tx.tx_date),  # חשבוניות ששולמו - לזיהוי תשלום כפול
    )
    return {"match": match}  # ההצעה או החריג -> לשדה match


def finish_tx_node(state: TxState) -> dict:
    """"שולחן האריזה": כל הדרכים מגיעות לכאן. אורז תוצאה אחת להעברה, ומוסיף לזיכרון הראשי."""
    return {
        "results": [
            TxResult(
                tx=state["tx"],  # ההעברה
                filter=state["filter"],  # תוצאת הסינון
                customer=state.get("customer"),  # תוצאת הזיהוי (None = לא הגענו)
                llm=state.get("llm"),  # הצעת המודל (None = לא הגענו)
                match=_final_match(state),  # ההחלטה הסופית: הצעה / חריג / None (סוננה)
            )
        ]
    }


# ---------- עזר ----------


def _final_match(state: TxState) -> InvoiceMatch | None:
    """עזר של finish_tx_node: ההחלטה הסופית על ההעברה, לפי הדרך שבה הגיעה."""
    outcome = state["filter"].outcome  # מה הסינון החליט
    if outcome == FilterOutcome.FILTERED:
        return None  # לא תקבול - אין הצעה ואין חריג
    if outcome == FilterOutcome.UNSUPPORTED:
        return _exception(ExceptionReason.UNSUPPORTED_TX_TYPE)
    if "match" in state:
        return state["match"]  # find_invoice כבר החליט
    llm = state.get("llm")  # הצעת המודל, אם הגענו אליו
    notes = (llm.reason,) if llm is not None else ()  # למשל "המודל לא היה זמין" / "אין ראיה תקפה"
    return _exception(ExceptionReason.CUSTOMER_NOT_IDENTIFIED, notes)


def _exception(reason: ExceptionReason, notes: tuple[str, ...] = ()) -> InvoiceMatch:
    """עזר של _final_match: חריג בלי לקוח ובלי חשבונית."""
    return InvoiceMatch(certainty=Certainty.EXCEPTION, exception_reason=reason, notes=notes)
