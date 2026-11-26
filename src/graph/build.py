"""
בניית הגרף (ה-49, ה-74): כאן מחברים את כל הצמתים והחצים לגרף אחד שאפשר להריץ.

אין כאן לוגיקה עסקית - רק שרטוט החיבורים:
הצמתים (העבודה) - מ-nodes.py; החצים עם החלטה - מ-routing.py; החצים הקבועים - כאן.

חלק 1 – תת-הגרף של העברה אחת. חלק 2 – הגרף הראשי של ריצת יום, ועזרי הפעלה.
"""

from __future__ import annotations

from datetime import date, datetime  # יום הבנק + שעת ההפעלה (למזהה הריצה)

from langchain_core.runnables import RunnableConfig  # סוג מילון ההגדרות של ריצה
from langgraph.checkpoint.base import BaseCheckpointSaver  # הסוג הכללי של שומר (קובץ / זיכרון בבדיקות)
from langgraph.graph import END, START, StateGraph  # בונה הגרף + צומתי הכניסה והיציאה
from langgraph.graph.state import CompiledStateGraph  # הסוג של גרף אחרי קומפילציה
from langgraph.types import Command, RetryPolicy  # הוראת חידוש + מדיניות ניסיונות חוזרים

from src.graph import nodes, routing  # הצמתים (העבודה) ופונקציות הניתוב (ההחלטות)
from src.graph.state import Decision, RunState, TxOutput, TxState  # הזיכרונות, סכמת הפלט, החלטה מהמסך
from src.graph.run_registry import RunRegistry, RunStatus  # טבלת הריצות + נעילה לפי run_id (ה-76)
from src.graph.toolbox import Toolbox  # ארגז הכלים שכל צומת מקבל

LLM_RETRY = RetryPolicy(  # מדיניות לצמתים שפונים למודל (ה-66)
    max_attempts=3,  # עד 3 ניסיונות בסך הכל (כולל הראשון)
    initial_interval=0.5,  # המתנה של חצי שנייה לפני הניסיון השני
    backoff_factor=2.0,  # כל המתנה כפולה מהקודמת: 0.5, ואז 1 שנייה
)
# אין timeout לצומת: ב-LangGraph הוא עובד רק בצמתים אסינכרוניים. ה-60 שניות - בבקשה עצמה (model.py), ה-66.


# ======================================================================
# חלק 1 – תת-הגרף: העברה אחת
# ======================================================================


def build_tx_graph() -> CompiledStateGraph:
    """תת-הגרף של העברה אחת (ה-49): סינון -> זיהוי -> [מודל -> אימות] -> חשבונית -> אריזה."""
    builder = StateGraph(
        TxState,  # הזיכרון הפנימי: tx, filter, customer, llm, match, results
        context_schema=Toolbox,  # כל צומת מקבל את ארגז הכלים (runtime.context)
        output_schema=TxOutput,  # מה שעולה לגרף הראשי: רק results (ה-74 סעיף 9)
    )

    # ----- הצמתים: שם בגרף <- הפונקציה שעושה את העבודה -----
    builder.add_node("filter_tx", nodes.filter_node)  # סינון: תקבול מלקוח?
    builder.add_node("identify_customer", nodes.identify_node)  # זיהוי לקוח בקוד
    builder.add_node(
        "llm_identify",  # זיהוי לקוח במודל
        nodes.llm_identify_node,  # הפונקציה שפונה למודל
        retry_policy=LLM_RETRY,  # עד 3 ניסיונות
        error_handler=nodes.llm_identify_failed,  # אחרי 3 כישלונות: גיבוי בטוח ("לא ידוע")
    )
    builder.add_node("verify_llm_customer", nodes.verify_node)  # אימות הצעת המודל בקוד
    builder.add_node("find_invoice", nodes.find_invoice_node)  # התאמת חשבונית + סיווג הפרש
    builder.add_node("finish_tx", nodes.finish_tx_node)  # אריזת תוצאה אחת

    # ----- החצים -----
    builder.add_edge(START, "filter_tx")  # כניסה: תמיד מתחילים בסינון
    builder.add_conditional_edges("filter_tx", routing.route_after_filter)  # תקבול / אריזה
    builder.add_conditional_edges("identify_customer", routing.route_after_identify)  # חשבונית / מודל / אריזה
    builder.add_edge("llm_identify", "verify_llm_customer")  # אחרי המודל תמיד אימות
    builder.add_conditional_edges("verify_llm_customer", routing.route_after_verify)  # חשבונית / אריזה
    builder.add_edge("find_invoice", "finish_tx")  # אחרי חשבונית תמיד אריזה
    builder.add_edge("finish_tx", END)  # יציאה: התוצאה עולה למעלה

    return builder.compile()  # בלי שמירה משלו - יורש את השמירה של הגרף הראשי


# ======================================================================
# חלק 2 – הגרף הראשי: ריצת יום
# ======================================================================


def build_run_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """הגרף הראשי של ריצת יום (ה-49, ה-71): טעינה -> פיצול -> איסוף -> הסברים -> מסך -> עצירה לאישור."""
    builder = StateGraph(
        RunState,  # הזיכרון הראשי: run_id, run_date, transactions, results...
        context_schema=Toolbox,  # אותו ארגז כלים - עובר אוטומטית גם לתת-הגרף
    )

    # ----- הצמתים -----
    builder.add_node("load_day", nodes.load_day_node)  # טעינת היום, בלי מה שכבר טופל
    builder.add_node("process_tx", build_tx_graph())  # תת-הגרף מחלק 1 - עותק לכל העברה
    builder.add_node("collect_results", nodes.collect_results_node)  # איסוף + בדיקת התנגשות
    builder.add_node(
        "explain_and_summarize",  # משפטים לחריגים + סיכום ריצה
        nodes.explain_node,  # הפונקציה שפונה למודל
        retry_policy=LLM_RETRY,  # עד 3 ניסיונות
        error_handler=nodes.explain_failed,  # אחרי 3 כישלונות: הכל מתבניות
    )
    builder.add_node("prepare_approval", nodes.prepare_approval_node)  # שורות המסך: לפני / מוחל / אחרי
    builder.add_node("human_approval", nodes.human_approval_node)  # רק עצירה + בדיקת ההחלטות
    builder.add_node("post_receipts", nodes.post_receipts_node)  # רישום + יומן ביקורת (שלב 7)

    # ----- החצים -----
    builder.add_edge(START, "load_day")  # כניסה: תמיד מתחילים בטעינה
    builder.add_conditional_edges(
        "load_day",  # אחרי הטעינה:
        routing.fan_out,  # Send לכל העברה, או ישר לאיסוף אם אין
        ["process_tx", "collect_results"],  # כל היעדים האפשריים
    )
    builder.add_edge("process_tx", "collect_results")  # כל העותקים -> איסוף אחד
    builder.add_edge("collect_results", "explain_and_summarize")  # -> הסברים
    builder.add_edge("explain_and_summarize", "prepare_approval")  # -> הכנת המסך
    builder.add_edge("prepare_approval", "human_approval")  # -> עצירה לאישור
    builder.add_edge("human_approval", "post_receipts")  # אחרי האישור - רישום
    builder.add_edge("post_receipts", END)  # סוף הריצה

    return builder.compile(checkpointer=checkpointer)  # השומר מחובר - תת-הגרף יורש אותו


# ======================================================================
# עזרי הפעלה
# ======================================================================


class RunNotWaitingError(Exception):
    """ניסיון לחדש ריצה שלא מחכה לאישור: כבר חודשה, הסתיימה, או לא קיימת."""


def make_run_id(run_date: date, started_at: datetime | None = None) -> str:
    """מזהה ריצה (ה-74 סעיף 2): run-<יום הבנק>-<תאריך ושעת ההפעלה>."""
    started = started_at or datetime.now()  # שעת ההפעלה; בבדיקות מעבירים שעה קבועה
    return f"run-{run_date.isoformat()}-{started:%Y%m%dT%H%M%S}"  # למשל run-2026-11-16-20261126T213000


def run_config(run_id: str) -> RunnableConfig:
    """הגדרות הריצה: המזהה לשומר + שם, תגיות ומטא-דאטה ל-LangSmith (ה-51)."""
    return {
        "configurable": {"thread_id": run_id},  # השומר: אותו מזהה = אותה ריצה
        "run_name": "bank-match-run",  # השם שיופיע ב-LangSmith
        "tags": ["bank-match"],  # תגית לסינון ריצות ב-LangSmith
        "metadata": {"run_id": run_id},  # חיפוש ריצה לפי המזהה ב-LangSmith
    }


def start_run(
    graph: CompiledStateGraph, toolbox: Toolbox, registry: RunRegistry, run_date: date, run_id: str
) -> dict:
    """מתחיל ריצה חדשה. רץ עד העצירה לאישור; אם עצרה - רושם אותה כ"מחכה". מחזיר את המצב (כולל "__interrupt__")."""
    config = run_config(run_id)  # המזהה לשומר + LangSmith
    result = graph.invoke(
        {"run_id": run_id, "run_date": run_date},  # הקלט: רק מזהה הריצה ויום הבנק
        config=config,
        context=toolbox,  # ארגז הכלים (לא נשמר בקובץ)
        durability="sync",  # כל נקודת שמירה לדיסק לפני הצעד הבא
    )
    if "human_approval" in graph.get_state(config).next:  # עצרה לאישור
        registry.mark_waiting(run_id)  # מעכשיו מותר לחדש - חידוש אחד בכל רגע
    return result


def resume_run(
    graph: CompiledStateGraph, toolbox: Toolbox, registry: RunRegistry, run_id: str, decisions: list[Decision]
) -> dict:
    """ממשיך ריצה שעצרה - רק מי שתפס את הנעילה (ה-76), ורק אם היא באמת מחכה לאישור (ה-75)."""
    registry.claim(run_id)  # שכבה 1: נעילה - בקשה שנייה במקביל מקבלת RunBusyError
    config = run_config(run_id)  # אותן הגדרות כמו בהתחלה
    try:
        snapshot = graph.get_state(config)  # נקודת השמירה האחרונה של הריצה, מהקובץ
        if "human_approval" not in snapshot.next:  # שכבה 2: בדיקה מול קובץ השמירה
            raise RunNotWaitingError(f"הריצה {run_id} לא מחכה לאישור - אולי כבר אושרה, או שאינה קיימת")
        nodes.validate_decisions(decisions, snapshot.values["approval_rows"])  # החלטה פגומה נעצרת כאן - לפני קובץ השמירה (ה-77)
        result = graph.invoke(
            Command(resume=decisions),  # התשובה ל-interrupt
            config=config,  # אותו מזהה -> אותה ריצה מהקובץ
            context=toolbox,  # ארגז הכלים - שוב, כי לא נשמר
            durability="sync",  # כל נקודת שמירה לדיסק לפני הצעד הבא
        )
    except BaseException:
        registry.release(run_id, RunStatus.WAITING)  # נכשל - משחררים את הנעילה, אפשר לנסות שוב
        raise
    still_waiting = "human_approval" in graph.get_state(config).next  # עצרה שוב? (למשל ההחלטות נדחו בבדיקה)
    registry.release(run_id, RunStatus.WAITING if still_waiting else RunStatus.DONE)  # שחרור הנעילה
    return result


def continue_posting(graph: CompiledStateGraph, toolbox: Toolbox, registry: RunRegistry, run_id: str) -> dict:
    """ממשיך ריצה שהרישום שלה נעצר בתקלה מערכתית (מסד נעול, דיסק): מריץ שוב את post_receipts.

    בטוח: מה שכבר נרשם מזוהה כ"כבר נרשמה" ולא נרשם שוב (ה-78).
    """
    registry.claim(run_id)  # אותה נעילה כמו בחידוש
    config = run_config(run_id)
    try:
        if "post_receipts" not in graph.get_state(config).next:  # הריצה לא נעצרה ברישום
            raise RunNotWaitingError(f"הריצה {run_id} לא נעצרה באמצע רישום")
        result = graph.invoke(None, config=config, context=toolbox, durability="sync")  # None = ממשיכים מהשמירה
    except BaseException:
        registry.release(run_id, RunStatus.WAITING)  # נכשל שוב - אפשר לנסות שוב
        raise
    registry.release(run_id, RunStatus.DONE)
    return result
