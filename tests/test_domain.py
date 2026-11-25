"""בדיקות לשכבת השפה המשותפת: כסף ומבני נתונים."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.domain.models import BankTransaction, Customer, FxRate, Invoice, OpenInvoice
from src.domain.money import Currency, Money, MoneyError, convert

# ---------- כסף ----------


def test_no_float_rounding_error():
    # 0.1 + 0.2 ב-float יוצא 0.30000000000000004. אצלנו - בדיוק 0.30
    assert Money.of("0.10", "ILS") + Money.of("0.20", "ILS") == Money.of("0.30", "ILS")


def test_float_is_rejected():
    with pytest.raises(MoneyError):
        Money.of(10.5, "ILS")


def test_too_many_digits_is_rejected_not_rounded():
    with pytest.raises(MoneyError):
        Money.of("10.005", "ILS")


def test_rounding_rule_half_up():
    assert Money.rounded(Decimal("2.345"), "ILS").amount == Decimal("2.35")
    assert Money.rounded(Decimal("2.344"), "ILS").amount == Decimal("2.34")


def test_cannot_mix_currencies():
    with pytest.raises(MoneyError):
        Money.of("10", "ILS") + Money.of("10", "USD")
    with pytest.raises(MoneyError):
        Money.of("10", "ILS") < Money.of("10", "USD")


def test_minor_units_round_trip():
    m = Money.of("5000.00", "ILS")
    assert m.to_minor() == 500000
    assert Money.from_minor(500000, "ILS") == m
    assert Money.from_minor(-30000, "USD") == Money.of("-300", "USD")


def test_withholding_scaled():
    # 10,000 ש"ח פחות 3% ניכוי במקור = 9,700
    assert Money.of("10000", "ILS").scaled(Decimal("0.97")) == Money.of("9700", "ILS")


def test_convert_usd_to_ils():
    assert convert(Money.of("1000", "USD"), Decimal("3.7012"), "ILS") == Money.of("3701.20", "ILS")


def test_unknown_currency():
    with pytest.raises(MoneyError):
        Money.of("1", "GBP")


# ---------- מבני נתונים ----------


def test_bank_transaction_from_csv_row():
    # בדיוק כמו שורה שנקראה מהקובץ: הכול מחרוזות, והעמודה נקראת date
    row = {
        "tx_id": "TX-00007",
        "date": "2026-11-09",
        "account_id": "ACC-05",
        "amount": "3172.00",
        "currency": "EUR",
        "bank_text": "EUROSHIP GMBH",
    }
    tx = BankTransaction.model_validate(row)
    assert tx.money == Money.of("3172", Currency.EUR)
    assert tx.is_incoming


def test_empty_invoice_id_rejected():
    with pytest.raises(ValidationError):
        Invoice.model_validate(
            {"invoice_id": "   ", "customer_id": "C001", "date": "2026-10-05",
             "amount": "100", "currency": "ILS"}
        )


def test_open_balance_above_amount_rejected():
    with pytest.raises(ValidationError):
        OpenInvoice.model_validate(
            {"invoice_id": "INV-10042", "customer_id": "C010", "date": "2026-09-17",
             "amount": "2200.00", "currency": "ILS", "open_balance": "3000.00"}
        )


def test_unexpected_column_rejected():
    with pytest.raises(ValidationError):
        Customer.model_validate(
            {"customer_id": "C001", "name": "x", "currency": "ILS",
             "withholding_rate": "0", "phone": "050"}
        )


def test_withholding_rate_range():
    with pytest.raises(ValidationError):
        Customer.model_validate(
            {"customer_id": "C001", "name": "x", "currency": "ILS", "withholding_rate": "3"}
        )


def test_no_rate_for_ils():
    with pytest.raises(ValidationError):
        FxRate.model_validate({"date": "2026-11-09", "currency": "ILS", "rate_to_ils": "1"})


def test_id_format_is_free():
    # פורמט אחר של מערכת אחרת (למשל NetSuite) עובר - אין תבנית קבועה
    c = Customer.model_validate(
        {"customer_id": "CUST001024", "name": "x", "currency": "ILS", "withholding_rate": "0"}
    )
    assert c.customer_id == "CUST001024"
