"""
מדיניות ההתאמה - כל הספים והכללים של סעיפים 4–8 במקום אחד.

המדיניות היא "חוברת הנהלים" - ערכים ולא פונקציות (זה לא ממשק).
אובייקט אחד שכל המנוע קורא ממנו, ולכן משנים כלל במקום אחד
(או בודקים סף אחר בניסוי) בלי לגעת בקוד ההתאמה.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from src.domain.money import Currency, Money


@dataclass(frozen=True)
class MatchPolicy:
    """הספים מסעיף 7 במדיניות + החלטות ה-52 עד ה-57."""

    # ---------- סינון (סעיף 4, ה-52) ----------
    internal_transfer_keywords: tuple[str, ...] = ("העברה בין חשבונות", "מסלול משלוחים")  # העברה פנימית / שם החברה -> מסוננת
    interest_keywords: tuple[str, ...] = ("ריבית",)  # ריבית -> מסוננת
    tax_refund_keywords: tuple[str, ...] = ("החזר מס", "רשות המסים")  # החזר מס -> מסוננת
    card_settlement_account: str = "ACC-04"  # חשבון הסליקה
    card_settlement_keyword: str = "סליקה"  # בחשבון הסליקה + המילה הזו -> חריג "לא נתמך"

    # ---------- זיהוי לקוח (סעיף 5, ה-53) ----------
    invoice_number_pattern: str = r"INV-\d{5}"  # איך נראה מספר חשבונית בטקסט הבנק
    fuzzy_accept: float = 90.0  # ציון דמיון שמספיק לזיהוי (סבירה)
    fuzzy_candidates: float = 75.0  # ציון שמספיק כדי להיות מועמד שנשלח למודל
    fuzzy_min_gap: float = 5.0  # שני לקוחות בהפרש קטן מזה -> לא בוחרים, שולחים למודל
    llm_quote_min_score: float = 70.0  # סימוכין מהמודל: דמיון מינימלי בין הציטוט לשם שבחר (ה-73)

    # ---------- מציאת חשבונית (סעיפים 6–8, ה-54 עד ה-57) ----------
    max_combination_size: int = 5  # עד כמה חשבוניות בצירוף אחד
    fee_limits: dict[Currency, Money] = field(  # עמלת בנק מקסימלית, לפי מטבע
        default_factory=lambda: {
            Currency.ILS: Money.of("25", Currency.ILS),  # העברה מקומית
            Currency.USD: Money.of("45", Currency.USD),  # בנק מתווך, דולר
            Currency.EUR: Money.of("40", Currency.EUR),  # בנק מתווך, אירו
        }
    )
    withholding_tolerance: Decimal = Decimal("1")  # סטייה מותרת בניכוי במקור (ביחידת מטבע)
    fx_match_tolerance: Decimal = Decimal("0.04")  # זיהוי חשבונית במט"ח: עד 4% מהצפוי (מרווח המרה של בנקים)
    fx_diff_tolerance: Decimal = Decimal("0.02")  # בקרה על הפרש שער: עד 2% = סבירה, מעל = חריג
