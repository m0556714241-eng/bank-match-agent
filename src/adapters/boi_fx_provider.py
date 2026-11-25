"""
שערים יציגים של בנק ישראל, עם מטמון בקובץ - מימוש של הממשק FxRateProvider.

כללים (החלטות ה-22, ה-34):
- השער היציג של אותו יום. יום בלי פרסום -> השער של יום הפרסום הקודם.
- בקשה אחת לטווח תאריכים לכל מטבע, והתוצאה נשמרת בקובץ המטמון (fx_rates.csv).
- בלי אינטרנט: עובד מהמטמון אם השער ידוע. אם לא - FxRateUnavailable (לא מנחשים).
בעולם האמיתי: טבלת השערים של NetSuite.
"""

from __future__ import annotations

import csv
import io
import os
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import requests

from src.adapters.interfaces import FxRateUnavailable
from src.domain.models import FxRate
from src.domain.money import Currency

# כתובת מאגר הסדרות של בנק ישראל (ממשק SDMX). סדרה לכל מטבע: RER_USD_ILS, RER_EUR_ILS
BOI_URL = "https://edge.boi.gov.il/FusionEdgeServer/sdmx/v2/data/dataflow/BOI.STATISTICS/EXR/1.0/RER_{cur}_ILS"

# כמה ימים אחורה מחפשים יום פרסום (סוף שבוע + חג ארוך)
DEFAULT_LOOKBACK_DAYS = 10

# פונקציה שמביאה טקסט CSV מבנק ישראל: (מטבע, מתאריך, עד תאריך) -> טקסט
Fetcher = Callable[[Currency, date, date], str]


def fetch_from_boi(currency: Currency, start: date, end: date) -> str:
    """פנייה אמיתית לבנק ישראל. מחזירה את התשובה כטקסט CSV."""
    response = requests.get(
        BOI_URL.format(cur=currency),
        params={"startPeriod": start.isoformat(), "endPeriod": end.isoformat(), "format": "csv"},
        timeout=15,  # לא מחכים לנצח אם האתר לא עונה
    )
    response.raise_for_status()  # תשובת שגיאה (404, 500...) -> חריגה
    return response.text


class BoiFxProvider:
    """שערים מבנק ישראל, עם מטמון בקובץ CSV."""

    def __init__(
        self,
        cache_path: Path,
        fetcher: Fetcher = fetch_from_boi,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self._cache_path = cache_path
        self._fetcher = fetcher
        self._lookback = timedelta(days=lookback_days)
        self._rates = _load_cache(cache_path)  # (מטבע, תאריך) -> FxRate
        self._fetched: set[tuple[Currency, date]] = set()  # מה כבר נשאל בריצה הזו

    def rate(self, currency: Currency, on: date) -> FxRate:
        """השער ליום. יום בלי פרסום - השער של יום הפרסום הקודם (עד lookback ימים אחורה)."""
        currency = Currency(currency)
        if currency == Currency.ILS:
            raise ValueError("אין שער לשקל מול שקל")

        # המטמון אמין לתאריך הזה רק אם כבר יש בו שערים מאותו יום או אחריו
        if not self._cache_covers(currency, on) and (currency, on) not in self._fetched:
            self._refresh(currency, on)

        found = self._latest_on_or_before(currency, on)
        if found is None:
            raise FxRateUnavailable(f"אין שער {currency} ליום {on} ובעשרת הימים שלפניו")
        return found

    # ---------- עזר ----------

    def _cache_covers(self, currency: Currency, on: date) -> bool:
        """עזר של rate: האם המטמון "יודע" על התאריך (יש בו שער מאותו יום או מאוחר יותר)."""
        dates = [d for (cur, d) in self._rates if cur == currency]
        return bool(dates) and max(dates) >= on

    def _refresh(self, currency: Currency, on: date) -> None:
        """עזר של rate: מביא מבנק ישראל את הטווח שחסר, ושומר במטמון. בלי אינטרנט - FxRateUnavailable."""
        self._fetched.add((currency, on))
        start = on - self._lookback
        try:
            text = self._fetcher(currency, start, on)
        except requests.RequestException as exc:
            raise FxRateUnavailable(f"אין חיבור לבנק ישראל והשער {currency} ליום {on} לא במטמון") from exc

        new_rates = _parse_boi_csv(text, currency)
        if new_rates:
            self._rates.update(new_rates)
            _save_cache(self._cache_path, self._rates)

    def _latest_on_or_before(self, currency: Currency, on: date) -> FxRate | None:
        """עזר של rate: השער האחרון שפורסם עד התאריך, בתוך חלון ה-lookback."""
        candidates = [
            r for (cur, d), r in self._rates.items() if cur == currency and on - self._lookback <= d <= on
        ]
        return max(candidates, key=lambda r: r.rate_date, default=None)


# ====================================================================
# קריאה וכתיבה של המטמון, ופענוח התשובה של בנק ישראל
# ====================================================================


def _load_cache(path: Path) -> dict[tuple[Currency, date], FxRate]:
    """עזר של __init__: קורא את קובץ המטמון. אין קובץ - מטמון ריק."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rates = [FxRate.model_validate(row) for row in csv.DictReader(f)]
    return {(r.currency, r.rate_date): r for r in rates}


def _save_cache(path: Path, rates: dict[tuple[Currency, date], FxRate]) -> None:
    """עזר של _refresh: כותב את כל המטמון לקובץ זמני, ורק אז מחליף - כדי שקובץ לא ייהרס באמצע."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "currency", "rate_to_ils"])
        for r in sorted(rates.values(), key=lambda r: (r.rate_date, r.currency)):
            writer.writerow([r.rate_date.isoformat(), r.currency, str(r.rate_to_ils)])
    os.replace(tmp, path)  # החלפה בפעולה אחת


def _parse_boi_csv(text: str, currency: Currency) -> dict[tuple[Currency, date], FxRate]:
    """עזר של _refresh: מפענח את ה-CSV של בנק ישראל - עמודת תאריך (TIME_PERIOD) ועמודת ערך (OBS_VALUE)."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    columns = {name.upper(): name for name in (reader.fieldnames or [])}
    if "TIME_PERIOD" not in columns or "OBS_VALUE" not in columns:
        raise FxRateUnavailable(f"תשובה לא צפויה מבנק ישראל. עמודות: {reader.fieldnames}")

    result: dict[tuple[Currency, date], FxRate] = {}
    for row in reader:
        raw_date, raw_value = row[columns["TIME_PERIOD"]], row[columns["OBS_VALUE"]]
        if not raw_value:
            continue  # יום בלי ערך
        try:
            rate = FxRate(rate_date=date.fromisoformat(raw_date), currency=currency, rate_to_ils=Decimal(raw_value))
        except (ValueError, InvalidOperation):
            continue  # שורה שלא ניתנת לפענוח - מדלגים, לא מנחשים
        result[(currency, rate.rate_date)] = rate
    return result
