"""
הצמתים של הגרף (ה-49, ה-74): עטיפות קצרות סביב הפונקציות שכבר כתבנו.

כל עטיפה: לוקחת מהזיכרון (state) ומארגז הכלים (runtime.context) את מה שהפונקציה צריכה,
קוראת לה, ומחזירה עדכון לשדה אחד בזיכרון. הפונקציות עצמן נשארות בקבצים שלהן.

חלק 1 – צמתי התת-גרף (העברה אחת). חלק 2 – צמתי הגרף הראשי (ריצה של יום).
"""

from __future__ import annotations

import json
from dataclasses import replace

from langgraph.errors import NodeError
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from src.adapters.interfaces import (
    Application,
    AuditEntry,
    Certainty,
    DuplicatePostingError,
    ExistingReceipt,
    LedgerStore,
    PostingControlError,
    ReceiptPosting,
)
from src.domain.money import Money
from src.engine.customer_id import identify_customer, normalize_name, verify_llm_customer
from src.engine.filters import filter_tx
from src.engine.invoice_match import find_invoice
from src.domain.models import BankTransaction, OpenInvoice
from src.engine.results import ExceptionReason, FilterOutcome, IdentifiedBy, InvoiceMatch
from src.graph.state import (
    ApprovalAction,
    ApprovalLine,
    ApprovalRow,
    Decision,
    PostingOutcome,
    PostingStatus,
    RunState,
    TxResult,
    TxState,
)
from src.graph.toolbox import Toolbox
from src.llm.customer_llm import fallback_suggestion, suggest_customer
from src.llm.explain import ExplainItem, compute_totals, explain_run, fallback_explanations

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


# ======================================================================
# חלק 2 – צמתי הגרף הראשי (ריצה של יום)
# ======================================================================


def load_day_node(state: RunState, runtime: Runtime[Toolbox]) -> dict:
    """טוען את ההעברות של היום, ומדלג על העברות שכבר טופלו בריצה קודמת (ה-71)."""
    tools = runtime.context  # ארגז הכלים
    loaded = tools.bank.load_day(state["run_date"])  # קובץ הבנק של היום: העברות + שגיאות + אזהרות
    ids = [tx.tx_id for tx in loaded.transactions]  # המזהים של כל ההעברות בקובץ
    done = tools.ledger.processed_tx_ids(ids)  # אילו מהן כבר טופלו (מהמסד)
    return {
        "transactions": [tx for tx in loaded.transactions if tx.tx_id not in done],  # רק מה שעוד לא טופל
        "intake_issues": [*loaded.intake_errors, *loaded.warnings],  # שגיאות ואזהרות קליטה - לדוח
        "customer_names": {c.customer_id: c.name for c in tools.ledger.customers()},  # מזהה -> שם
    }


def collect_results_node(state: RunState, runtime: Runtime[Toolbox]) -> dict:
    """איסוף התוצאות + בדיקת התנגשות: שתי העברות לא יקבלו הצעה לאותה חשבונית מעבר ליתרה (ה-71)."""
    checked = _resolve_conflicts(
        state.get("results", []),  # התוצאות של כל ההעברות (נאספו בכלל האיחוד)
        runtime.context.ledger,  # המסד - ליתרות הפתוחות
    )
    return {"checked_results": checked}  # התוצאות אחרי הבדיקה -> לשדה checked_results


def explain_node(state: RunState, runtime: Runtime[Toolbox]) -> dict:
    """משפט לכל שורה וסיכום ריצה: הצעות מתבנית, חריגים וסיכום מהמודל + בדיקת מספרים (ה-69)."""
    tools = runtime.context  # ארגז הכלים
    items, filtered = _explain_inputs(state)  # השורות להסבר + כמה סוננו
    result = explain_run(
        tools.explain_model,  # המודל להסברים (Claude, 4,000 טוקנים)
        items,  # השורות
        compute_totals(items, filtered),  # מספרי הסיכום - הקוד סופר
        tools.policy,  # חוברת הנהלים (ספים מותרים במשפטים)
    )
    return {"explanations": result}  # המשפטים והסיכום -> לשדה explanations


def explain_failed(state: RunState, error: NodeError) -> Command:
    """תוכנית החירום של צומת ההסברים: כל הניסיונות נכשלו ← הכל מתבניות ← ממשיכים (ה-66)."""
    items, filtered = _explain_inputs(state)  # אותן שורות
    return Command(
        update={"explanations": fallback_explanations(items, compute_totals(items, filtered))},  # הכל מתבניות
        goto="prepare_approval",  # ממשיכים להכנת מסך האישור
    )


def prepare_approval_node(state: RunState, runtime: Runtime[Toolbox]) -> dict:
    """מכין את שורות מסך האישור: לכל העברה - המשפט, ולכל חשבונית - יתרה לפני / מוחל / אחרי (ה-74)."""
    rows = _approval_rows(
        state["checked_results"],  # התוצאות אחרי בדיקת ההתנגשות
        state["explanations"].sentences,  # המשפט לכל העברה
        state["customer_names"],  # מזהה -> שם
        runtime.context.ledger,  # המסד - ליתרות הפתוחות
    )
    return {"approval_rows": rows}  # שורות המסך -> לשדה approval_rows


def human_approval_node(state: RunState) -> dict:
    """עצירה לאישור אדם. בלי שום פעולה מלבד העצירה - כי בחידוש הצומת רץ שוב מההתחלה (ה-71)."""
    answer = interrupt(
        {
            "run_id": state["run_id"],  # מזהה הריצה
            "summary": state["explanations"].summary,  # סיכום הריצה - בראש המסך
            "rows": state["approval_rows"],  # השורות לאישור
        }
    )  # הגרף נעצר כאן ונשמר לקובץ. בחידוש - interrupt מחזיר את ההחלטות מהמסך
    return {"decisions": validate_decisions(answer, state["approval_rows"])}  # קו הגנה שני - resume_run כבר בדק  # ההחלטות, אחרי בדיקה


# ---------- עזר לחלק 2 ----------


def _resolve_conflicts(results: list[TxResult], ledger: LedgerStore) -> list[TxResult]:
    """עזר של collect_results_node: עובר על ההצעות מהוותיקה לחדשה; הצעה שחורגת מהיתרה שנשארה - הופכת לחריג."""
    remaining: dict[str, Money] = {}  # חשבונית -> כמה נשאר בה אחרי ההצעות שכבר התקבלו
    taken_by: dict[str, str] = {}  # חשבונית -> ההעברה הראשונה שקיבלה עליה הצעה
    checked: list[TxResult] = []
    for result in sorted(results, key=lambda r: (r.tx.tx_date, r.tx.tx_id)):  # הוותיקה קודם
        match = result.match
        if match is None or match.is_exception or not match.applications:
            checked.append(result)  # סוננה / חריג - אין מה לבדוק
            continue
        open_now = {inv.invoice_id: inv.open_money for inv in ledger.open_invoices(as_of=result.tx.tx_date)}
        for app in match.applications:
            remaining.setdefault(app.invoice_id, open_now[app.invoice_id])  # פעם ראשונה - היתרה מהמסד
        clash = next((app for app in match.applications if app.amount > remaining[app.invoice_id]), None)
        if clash is not None:  # אין מספיק יתרה - חשבונית כבר נלקחה בריצה הזו
            conflict = InvoiceMatch(
                certainty=Certainty.EXCEPTION,
                customer_id=match.customer_id,
                exception_reason=ExceptionReason.INVOICE_CONFLICT,
                evidence={"invoice": clash.invoice_id, "other_tx": taken_by.get(clash.invoice_id, "-")},
            )
            checked.append(replace(result, match=conflict))
            continue
        for app in match.applications:
            remaining[app.invoice_id] = remaining[app.invoice_id] - app.amount  # מורידים את מה שהוצע
            taken_by.setdefault(app.invoice_id, result.tx.tx_id)
        checked.append(result)
    return checked


def _explain_inputs(state: RunState) -> tuple[list[ExplainItem], int]:
    """עזר של explain_node ו-explain_failed: השורות להסבר (רק מה שאינו מסונן), וכמה סוננו."""
    names = state["customer_names"]  # מזהה -> שם
    results = state["checked_results"]  # התוצאות אחרי בדיקת ההתנגשות
    items = [
        ExplainItem(r.tx, r.match, names.get(r.match.customer_id or ""))  # העברה + החלטה + שם הלקוח
        for r in results
        if r.match is not None
    ]
    return items, sum(1 for r in results if r.match is None)


def _approval_rows(
    results: list[TxResult],
    sentences: dict[str, str],
    names: dict[str, str],
    ledger: LedgerStore,
) -> list[ApprovalRow]:
    """עזר של prepare_approval_node: שורה לכל העברה שאינה מסוננת, עם יתרה לפני/אחרי לכל חשבונית."""
    remaining: dict[str, Money] = {}  # חשבונית -> יתרה אחרי ההצעות שכבר הוצגו (אותו סדר כמו בבדיקת ההתנגשות)
    rows: list[ApprovalRow] = []
    for result in results:
        match = result.match
        if match is None:
            continue  # סוננה - לא מוצגת לאישור
        lines: list[ApprovalLine] = []
        actions = _allowed_actions(match)  # אילו כפתורים יוצגו
        if not match.is_exception:
            open_now = {inv.invoice_id: inv.open_money for inv in ledger.open_invoices(as_of=result.tx.tx_date)}
            for app in match.applications:
                before = remaining.get(app.invoice_id, open_now[app.invoice_id])  # היתרה לפני
                after = before - app.amount  # היתרה אחרי
                remaining[app.invoice_id] = after
                lines.append(ApprovalLine(app.invoice_id, before, app.amount, after))
        elif ApprovalAction.APPLY_TO_OLDEST in actions:  # חלקי בלי מספר: מראים מראש על איזו חשבונית (ה-78)
            oldest = oldest_open_invoice(ledger, result.tx, match.customer_id)
            if oldest is not None and result.tx.money <= oldest.open_money:  # לא חורגים מהיתרה
                lines.append(ApprovalLine(oldest.invoice_id, oldest.open_money, result.tx.money,
                                          oldest.open_money - result.tx.money))
            else:
                actions = (ApprovalAction.REJECT,)  # אין חשבונית מתאימה - רק טיפול ידני
        rows.append(
            ApprovalRow(
                tx_id=result.tx.tx_id,  # מזהה ההעברה
                certainty=match.certainty,  # ודאית / סבירה / חריג
                customer_name=names.get(match.customer_id or ""),  # שם הלקוח (None = לא זוהה)
                bank_amount=result.tx.money,  # הסכום בבנק
                sentence=sentences.get(result.tx.tx_id, ""),  # המשפט מצומת ההסברים
                lines=tuple(lines),  # החשבוניות: לפני / מוחל / אחרי
                adjustments=match.adjustments,  # עמלה / ניכוי / הפרש שער
                allowed_actions=actions,  # אילו כפתורים יוצגו
            )
        )
    return rows


def _allowed_actions(match: InvoiceMatch) -> tuple[ApprovalAction, ...]:
    """עזר של _approval_rows: הצעה - לאשר/לדחות; חלקי בלי מספר - להחיל על הוותיקה / לדחות; חריג אחר - לטיפול ידני."""
    if not match.is_exception:
        return (ApprovalAction.APPROVE, ApprovalAction.REJECT)
    if match.exception_reason == ExceptionReason.PARTIAL_WITHOUT_REFERENCE:
        return (ApprovalAction.APPLY_TO_OLDEST, ApprovalAction.REJECT)  # ה-57; זכות ללקוח ירדה (ה-78)
    return (ApprovalAction.REJECT,)  # חריג אחר: לא נרשם אוטומטית - לטיפול ידני


def oldest_open_invoice(ledger: LedgerStore, tx: BankTransaction, customer_id: str | None) -> OpenInvoice | None:
    """עזר של _approval_rows ושל הרישום: החשבונית הפתוחה הוותיקה של הלקוח, במטבע ההעברה (None = אין)."""
    if customer_id is None:
        return None
    same_currency = [
        inv for inv in ledger.open_invoices(as_of=tx.tx_date, customer_id=customer_id)  # פתוחות ביום ההעברה
        if inv.currency == tx.currency  # רק במטבע של ההעברה
    ]
    return min(same_currency, key=lambda inv: (inv.issue_date, inv.invoice_id), default=None)  # הוותיקה


def validate_decisions(answer: object, rows: list[ApprovalRow]) -> list[Decision]:
    """עזר של human_approval_node ושל resume_run (build.py): מה שחזר מהמסך תקין? העברה מוכרת, פעם אחת, ופעולה שמותרת לה.

    resume_run קורא לה לפני החידוש, כדי שהחלטה פגומה לא תישמר בקובץ השמירה ותיתקע שם (ה-77)."""
    if not isinstance(answer, list) or not all(isinstance(d, Decision) for d in answer):
        raise ValueError("ההחלטות מהמסך חייבות להיות רשימה של Decision")
    allowed = {row.tx_id: row.allowed_actions for row in rows}  # העברה -> הפעולות המותרות לה
    seen: set[str] = set()
    for d in answer:
        if d.tx_id not in allowed:
            raise ValueError(f"החלטה על העברה שלא הוצגה: {d.tx_id}")
        if d.tx_id in seen:
            raise ValueError(f"שתי החלטות על אותה העברה: {d.tx_id}")
        if d.action not in allowed[d.tx_id]:
            raise ValueError(f"הפעולה {d.action} לא מותרת להעברה {d.tx_id}")
        seen.add(d.tx_id)
    return list(answer)  # העברה בלי החלטה - לא נרשמת בריצה הזו


# ======================================================================
# חלק 3 – הרישום (שלב 7, ה-78)
# ======================================================================

ALIAS_SOURCES = {IdentifiedBy.ALIAS, IdentifiedBy.FUZZY, IdentifiedBy.LLM}  # מאלה לומדים כינוי (לא ממספר חשבונית / שם זהה)


def post_receipts_node(state: RunState, runtime: Runtime[Toolbox]) -> dict:
    """רושם את מה שאושר, מעביר לטיפול ידני את מה שנדחה, וכותב יומן ביקורת לכל החלטה.

    בעיה עסקית בהעברה אחת -> סטטוס, והשאר ממשיכות. כל שגיאה אחרת (מסד, דיסק) -> עוצרת את הצומת;
    אפשר להריץ אותו שוב בבטחה (continue_posting), כי הרישום אידמפוטנטי.
    """
    tools = runtime.context  # ארגז הכלים
    results = {r.tx.tx_id: r for r in state["checked_results"]}  # העברה -> התוצאה שלה
    decided = {d.tx_id: d for d in state.get("decisions", [])}  # העברה -> ההחלטה (מי שלא הוחלט - חסר)
    outcomes: list[PostingOutcome] = []
    for row in state["approval_rows"]:  # כל שורה שהוצגה במסך, לפי הסדר
        decision = decided.get(row.tx_id)
        if decision is None:
            outcomes.append(PostingOutcome(row.tx_id, PostingStatus.PENDING))  # לא נוגעים - תוצע שוב
            continue
        outcome = _apply_decision(decision, row, results[row.tx_id], state["run_id"], tools.ledger)
        tools.ledger.record_audit(  # יומן ביקורת: מה הוצג, מה הוחלט, מי, ומה קרה
            AuditEntry(
                run_id=state["run_id"],
                tx_id=row.tx_id,
                proposal=_proposal_json(row),
                action=decision.action,
                approved_by=decision.approved_by,
                outcome=outcome.status,
                receipt_id=outcome.receipt_id,
                detail=outcome.detail,
            )
        )
        outcomes.append(outcome)
    return {"postings": outcomes}  # התוצאות -> לשדה postings (לדוח)


def _apply_decision(
    decision: Decision, row: ApprovalRow, result: TxResult, run_id: str, ledger: LedgerStore
) -> PostingOutcome:
    """עזר של post_receipts_node: מבצע החלטה אחת ומחזיר מה קרה."""
    tx_id = row.tx_id
    if decision.action == ApprovalAction.REJECT:  # לטיפול ידני
        try:
            ledger.mark_processed(tx_id, run_id, "MANUAL", f"נדחתה על ידי {decision.approved_by}")
        except DuplicatePostingError:
            pass  # כבר סומנה (למשל הצומת רץ שוב אחרי תקלה) - אותה תוצאה
        return PostingOutcome(tx_id, PostingStatus.MANUAL, detail="לטיפול ידני")

    posting = _posting_for(decision, row, result, run_id)  # מה בדיוק אושר
    existing = ledger.receipt_for_tx(tx_id)
    if existing is not None:  # כבר רשום - לוודא שזה אותו דבר לפני שמדלגים
        return _compare_existing(posting, existing)

    changed = _changed_since_shown(row, result, decision, ledger)
    if changed:
        return PostingOutcome(tx_id, PostingStatus.CHANGED_SINCE_SHOWN, detail=changed)

    try:
        receipt = ledger.post_receipt(posting)  # הכל או כלום + בקרת יתרה אחרי (קיים מ-שלב 3)
    except DuplicatePostingError:  # מישהו רשם בדיוק עכשיו
        again = ledger.receipt_for_tx(tx_id)
        if again is None:
            return PostingOutcome(tx_id, PostingStatus.CONFLICTS_WITH_EXISTING, detail="ההעברה כבר סומנה כמטופלת")
        return _compare_existing(posting, again)
    except PostingControlError as exc:
        return PostingOutcome(tx_id, PostingStatus.CONTROL_FAILED, detail=str(exc))

    note = _learn_alias(result, posting.customer_id, ledger)  # כל אישור מלמד את הסוכן
    return PostingOutcome(tx_id, PostingStatus.POSTED, receipt_id=receipt.receipt_id, detail=note)


def _posting_for(decision: Decision, row: ApprovalRow, result: TxResult, run_id: str) -> ReceiptPosting:
    """עזר של _apply_decision: בקשת הרישום - בדיוק מה שהוצג במסך."""
    match = result.match
    if decision.action == ApprovalAction.APPLY_TO_OLDEST:  # החשבונית שהוצגה בשורה
        line = row.lines[0]
        return ReceiptPosting(
            tx_id=row.tx_id, customer_id=match.customer_id, amount=line.applied, certainty=match.certainty,
            approved_by=decision.approved_by, run_id=run_id, applications=[Application(line.invoice_id, line.applied)],
        )
    return ReceiptPosting(  # APPROVE: ההצעה כמו שהיא
        tx_id=row.tx_id, customer_id=match.customer_id, amount=match.receipt_amount, certainty=match.certainty,
        approved_by=decision.approved_by, run_id=run_id,
        applications=list(match.applications), adjustments=list(match.adjustments),
    )


def _changed_since_shown(row: ApprovalRow, result: TxResult, decision: Decision, ledger: LedgerStore) -> str:
    """עזר של _apply_decision: בקרה לפני רישום - היתרה עכשיו שווה למה שהוצג? ("" = הכל תקין)."""
    open_now = {inv.invoice_id: inv.open_money for inv in ledger.open_invoices(as_of=result.tx.tx_date)}
    for line in row.lines:
        now = open_now.get(line.invoice_id)  # None = החשבונית כבר סגורה
        if now != line.open_before:
            return f"היתרה של {line.invoice_id} הוצגה {line.open_before}, ועכשיו {now if now is not None else 'סגורה'}"
    if decision.action == ApprovalAction.APPLY_TO_OLDEST:  # היעד לא משתנה בשקט
        oldest = oldest_open_invoice(ledger, result.tx, result.match.customer_id)
        if oldest is None or oldest.invoice_id != row.lines[0].invoice_id:
            return f"החשבונית הוותיקה כבר אינה {row.lines[0].invoice_id}"
    return ""


def _compare_existing(posting: ReceiptPosting, existing: ExistingReceipt) -> PostingOutcome:
    """עזר של _apply_decision: תקבול שכבר רשום - זהה למה שאושר (מדלגים) או שונה (לטיפול ידני)?"""
    same = (
        existing.customer_id == posting.customer_id
        and existing.amount == posting.amount
        and sorted((a.invoice_id, a.amount.amount) for a in existing.applications)
        == sorted((a.invoice_id, a.amount.amount) for a in posting.applications)
    )
    if same:
        return PostingOutcome(posting.tx_id, PostingStatus.ALREADY_POSTED, receipt_id=existing.receipt_id)
    return PostingOutcome(
        posting.tx_id, PostingStatus.CONFLICTS_WITH_EXISTING, receipt_id=existing.receipt_id,
        detail=f"רשום כבר תקבול {existing.receipt_id} שונה ממה שאושר",
    )


def _learn_alias(result: TxResult, customer_id: str, ledger: LedgerStore) -> str:
    """עזר של _apply_decision: אחרי אישור - לומדים את טקסט הבנק ככינוי של הלקוח (סעיף 12, ה-78)."""
    if result.customer is None or result.customer.identified_by not in ALIAS_SOURCES:
        return ""  # זוהה לפי מספר חשבונית / שם זהה - אין מה ללמוד
    bank_name = result.tx.bank_text.strip()
    if not bank_name:
        return ""
    owners = {a.customer_id for a in ledger.aliases() if normalize_name(a.bank_name) == normalize_name(bank_name)}
    if owners and customer_id not in owners:  # השם כבר שייך ללקוח אחר - לא מערבבים
        return f"כינוי לא נלמד: '{bank_name}' כבר משויך ל-{', '.join(sorted(owners))}"
    ledger.confirm_alias(bank_name, customer_id)
    return f"כינוי נלמד: '{bank_name}'"


def _proposal_json(row: ApprovalRow) -> str:
    """עזר של post_receipts_node: ההצעה שהוצגה במסך, כ-JSON קריא, ליומן הביקורת."""
    return json.dumps(
        {
            "certainty": row.certainty,
            "customer": row.customer_name,
            "bank_amount": str(row.bank_amount),
            "lines": [[ln.invoice_id, str(ln.open_before), str(ln.applied), str(ln.open_after)] for ln in row.lines],
            "adjustments": [[adj.type, str(adj.amount)] for adj in row.adjustments],
            "sentence": row.sentence,
        },
        ensure_ascii=False,  # עברית נשמרת כעברית
    )
