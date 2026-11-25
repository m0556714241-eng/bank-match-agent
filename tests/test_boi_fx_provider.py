"""בדיקות למתאם השערים - עם "בנק ישראל מדומה", בלי אינטרנט."""

from datetime import date
from decimal import Decimal

import pytest
import requests

from src.adapters.boi_fx_provider import BoiFxProvider
from src.adapters.interfaces import FxRateUnavailable
from src.domain.money import Currency

USD = Currency.USD

# תשובה בפורמט של בנק ישראל: חמישי 12/11 ושישי 13/11 פורסמו, שבת וראשון לא
BOI_RESPONSE = (
    "STRUCTURE,STRUCTURE_ID,SERIES_CODE,TIME_PERIOD,OBS_VALUE\n"
    "dataflow,BOI.STATISTICS:EXR(1.0),RER_USD_ILS,2026-11-12,3.7012\n"
    "dataflow,BOI.STATISTICS:EXR(1.0),RER_USD_ILS,2026-11-13,3.7150\n"
)


class FakeBoi:
    """בנק ישראל מדומה: מחזיר תשובה קבועה וסופר כמה פעמים פנו אליו."""

    def __init__(self, text=BOI_RESPONSE, fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def __call__(self, currency, start, end):
        self.calls += 1
        if self.fail:
            raise requests.ConnectionError("no internet")
        return self.text


def test_rate_on_publication_day(tmp_path):
    fx = BoiFxProvider(tmp_path / "fx.csv", fetcher=FakeBoi())
    r = fx.rate(USD, date(2026, 11, 12))
    assert r.rate_to_ils == Decimal("3.7012")


def test_weekend_uses_previous_publication(tmp_path):
    fx = BoiFxProvider(tmp_path / "fx.csv", fetcher=FakeBoi())
    r = fx.rate(USD, date(2026, 11, 15))  # ראשון - בלי פרסום
    assert r.rate_date == date(2026, 11, 13)  # השער של שישי, והתאריך האמיתי שלו נשמר


def test_cache_is_saved_and_reused_offline(tmp_path):
    path = tmp_path / "fx.csv"
    BoiFxProvider(path, fetcher=FakeBoi()).rate(USD, date(2026, 11, 13))
    assert path.exists()
    # ריצה חדשה בלי אינטרנט: השער כבר במטמון
    offline = FakeBoi(fail=True)
    r = BoiFxProvider(path, fetcher=offline).rate(USD, date(2026, 11, 12))
    assert r.rate_to_ils == Decimal("3.7012")
    assert offline.calls == 0


def test_offline_and_not_cached_is_unavailable(tmp_path):
    fx = BoiFxProvider(tmp_path / "fx.csv", fetcher=FakeBoi(fail=True))
    with pytest.raises(FxRateUnavailable):
        fx.rate(USD, date(2026, 11, 12))


def test_one_request_per_day_in_a_run(tmp_path):
    boi = FakeBoi()
    fx = BoiFxProvider(tmp_path / "fx.csv", fetcher=boi)
    for _ in range(5):  # חמש תנועות דולר באותו יום
        fx.rate(USD, date(2026, 11, 15))
    assert boi.calls == 1


def test_no_rate_in_window_is_unavailable(tmp_path):
    fx = BoiFxProvider(tmp_path / "fx.csv", fetcher=FakeBoi(), lookback_days=1)
    with pytest.raises(FxRateUnavailable):
        fx.rate(USD, date(2026, 11, 20))  # שבוע אחרי הפרסום האחרון


def test_unexpected_response_is_unavailable(tmp_path):
    fx = BoiFxProvider(tmp_path / "fx.csv", fetcher=FakeBoi(text="<html>error</html>"))
    with pytest.raises(FxRateUnavailable):
        fx.rate(USD, date(2026, 11, 12))
