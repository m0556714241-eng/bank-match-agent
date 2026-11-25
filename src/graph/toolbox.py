"""
ארגז הכלים של הגרף (ה-71, ה-74): כל מה שהצמתים צריכים ואינו נתון -
המסד, מקור הבנק, ספק השערים, המדיניות והמודלים.

לא נשמר במצב: נמסר לגרף בהפעלה (context של LangGraph).
בבדיקות מחליפים כאן מסד זמני ומודל מדומה - בלי לגעת בגרף.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.adapters.boi_fx_provider import BoiFxProvider
from src.adapters.csv_bank_source import CsvBankSource
from src.adapters.interfaces import BankSource, FxRateProvider, LedgerStore
from src.adapters.sqlite_ledger import SqliteLedger
from src.config import Settings
from src.engine.policy import MatchPolicy
from src.llm import customer_llm, explain
from src.llm.customer_llm import StructuredChatModel
from src.llm.model import make_chat_model


@dataclass(frozen=True)
class Toolbox:
    """הכלים - כל שדה הוא ממשק, לא מימוש (ארכיטקטורה נקייה)."""

    ledger: LedgerStore  # המסד: לקוחות, חשבוניות, כינויים, רישום
    bank: BankSource  # קבצי הבנק היומיים
    fx: FxRateProvider  # שערי בנק ישראל
    policy: MatchPolicy  # חוברת הנהלים: כל הספים
    identify_model: StructuredChatModel  # המודל לזיהוי לקוח (תקרה 300 טוקנים)
    explain_model: StructuredChatModel  # המודל להסברים (תקרה 4,000 טוקנים)


def build_toolbox(settings: Settings) -> Toolbox:              # מקבלת את ההגדרות, מחזירה ארגז מלא
    """בונה את ארגז הכלים האמיתי מההגדרות. בבדיקות לא משתמשים בה - בונים Toolbox עם כלים מדומים."""
    return Toolbox(                                            # יוצרת ארגז, תא אחרי תא:
        ledger=SqliteLedger(settings.db_path),                 # המסד: פותח את הקובץ ledger.db
        bank=CsvBankSource(                                    # קורא קבצי הבנק:
            settings.bank_dir,                                 #   מאיזו תיקייה לקרוא
            date_tolerance_days=settings.date_tolerance_days,  #   כמה ימים אחורה תאריך נחשב תקין (4, ה-46)
        ),
        fx=BoiFxProvider(                                      # ספק השערים:
            settings.fx_cache_csv,                             #   קובץ המטמון של השערים
            lookback_days=settings.fx_lookback_days,           #   כמה ימים אחורה לחפש שער שפורסם (10)
        ),
        policy=MatchPolicy(),                                  # חוברת הנהלים, עם ערכי ברירת המחדל
        identify_model=make_chat_model(                        # מודל לזיהוי לקוח:
            settings,                                          #   המפתח ושם המודל מ-.env
            max_tokens=customer_llm.MAX_TOKENS,                #   תקרה: 300 טוקנים (תשובה קצרה)
        ),
        explain_model=make_chat_model(                         # מודל להסברים:
            settings,                                          #   אותו מפתח ואותו מודל
            max_tokens=explain.MAX_TOKENS,                     #   תקרה: 4,000 טוקנים (כל החריגים + סיכום)
        ),
    )
