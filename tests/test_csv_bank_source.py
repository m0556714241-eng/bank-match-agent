"""בדיקות למתאם קבצי הבנק: קליטה תקינה, רעש של Excel, ושגיאות קליטה."""

from datetime import date
from decimal import Decimal

from src.adapters.interfaces import IntakeErrorCode, IntakeWarningCode
from src.adapters.csv_bank_source import CsvBankSource

HEADER = "tx_id,date,account_id,amount,currency,bank_text\n"
DAY = date(2026, 11, 9)


def _write(tmp_path, body: str):
    # כותב קובץ בדיוק כמו ש-Excel שומר: עם BOM בהתחלה
    path = tmp_path / f"bank_transactions_{DAY.isoformat()}.csv"
    path.write_text(HEADER + body, encoding="utf-8-sig")
    return CsvBankSource(tmp_path)


def test_valid_rows(tmp_path):
    src = _write(tmp_path, "TX-00007,2026-11-09,ACC-05,3172.00,EUR,EUROSHIP GMBH\n")
    result = src.load_day(DAY)
    assert len(result.transactions) == 1
    assert result.transactions[0].amount == Decimal("3172.00")
    assert result.intake_errors == []


def test_excel_noise_is_rounded_with_warning(tmp_path):
    src = _write(tmp_path, "TX-00008,2026-11-09,ACC-01,1890.0000000001,ILS,פייפלו\n")
    result = src.load_day(DAY)
    assert result.transactions[0].amount == Decimal("1890.00")
    assert [w.code for w in result.warnings] == [IntakeWarningCode.AMOUNT_ROUNDED]
    assert result.warnings[0].row_number == 2


def test_trailing_zeros_ok(tmp_path):
    src = _write(tmp_path, "TX-00009,2026-11-09,ACC-01,4200.000,ILS,x\n")
    result = src.load_day(DAY)
    assert result.transactions[0].amount == Decimal("4200.00")
    assert result.warnings == []


def test_real_third_digit_is_intake_error(tmp_path):
    src = _write(
        tmp_path,
        "TX-00010,2026-11-09,ACC-01,1890.375,ILS,x\n"
        "TX-00011,2026-11-09,ACC-01,500.00,ILS,y\n",
    )
    result = src.load_day(DAY)
    # השורה הפגומה לא נקלטה - ושאר הקובץ המשיך
    assert [t.tx_id for t in result.transactions] == ["TX-00011"]
    assert result.intake_errors[0].code == IntakeErrorCode.INVALID_AMOUNT_PRECISION
    assert result.intake_errors[0].row_number == 2


def test_each_error_code(tmp_path):
    src = _write(
        tmp_path,
        "TX-00020,2026-11-09,ACC-01,abc,ILS,x\n"  # סכום לא מספר
        "TX-00021,2026-11-09,ACC-01,100,GBP,x\n"  # מטבע לא מוכר
        "TX-00022,2026-13-40,ACC-01,100,ILS,x\n"  # תאריך לא תקין
        "TX-00023,2026-11-09,,100,ILS,x\n"  # שדה חובה ריק
        "TX-00024,2026-11-09,ACC-01,100,ILS,x\n"
        "TX-00024,2026-11-09,ACC-01,100,ILS,x\n"  # כפילות
        "TX-00025,2026-11-09,ACC-01,0,ILS,x\n",  # סכום אפס
    )
    codes = [e.code for e in src.load_day(DAY).intake_errors]
    assert codes == [
        IntakeErrorCode.INVALID_AMOUNT,
        IntakeErrorCode.UNKNOWN_CURRENCY,
        IntakeErrorCode.INVALID_DATE,
        IntakeErrorCode.MISSING_FIELD,
        IntakeErrorCode.DUPLICATE_TX_ID,
        IntakeErrorCode.INVALID_AMOUNT,
    ]


def test_changed_header_loads_nothing(tmp_path):
    path = tmp_path / f"bank_transactions_{DAY.isoformat()}.csv"
    path.write_text("tx_id,date,amount\nTX-1,2026-11-09,5\n", encoding="utf-8")
    result = CsvBankSource(tmp_path).load_day(DAY)
    assert result.transactions == []
    assert len(result.intake_errors) == 1


def test_missing_file_is_warning(tmp_path):
    result = CsvBankSource(tmp_path).load_day(DAY)
    assert result.transactions == []
    assert result.warnings[0].code == IntakeWarningCode.FILE_MISSING
    assert result.warnings[0].row_number is None


def test_empty_bank_text_allowed(tmp_path):
    src = _write(tmp_path, "TX-00030,2026-11-09,ACC-01,100,ILS,\n")
    assert src.load_day(DAY).transactions[0].bank_text == ""


def test_weekend_date_is_silent(tmp_path):
    # העברה משישי (06/11) שמופיעה בקובץ של ראשון (09/11) - רגיל, בלי אזהרה
    src = _write(tmp_path, "TX-00040,2026-11-06,ACC-01,100,ILS,x\n")
    result = src.load_day(DAY)
    assert len(result.transactions) == 1
    assert result.warnings == []


def test_old_date_warns(tmp_path):
    src = _write(tmp_path, "TX-00041,2026-11-02,ACC-01,100,ILS,x\n")
    result = src.load_day(DAY)
    assert len(result.transactions) == 1  # נקלטת בכל זאת
    assert [w.code for w in result.warnings] == [IntakeWarningCode.DATE_OUT_OF_RANGE]


def test_future_date_warns(tmp_path):
    src = _write(tmp_path, "TX-00042,2026-11-10,ACC-01,100,ILS,x\n")
    result = src.load_day(DAY)
    assert [w.code for w in result.warnings] == [IntakeWarningCode.DATE_OUT_OF_RANGE]
