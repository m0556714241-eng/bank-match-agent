"""בדיקות למנוע ההתאמה: כל צומת וכל כלל בנפרד, על נתונים קטנים."""

from datetime import date
from decimal import Decimal

from src.adapters.interfaces import AdjustmentType, Certainty, FxRateUnavailable
from src.domain.models import Alias, BankTransaction, Customer, FxRate, Invoice, OpenInvoice
from src.domain.money import Currency, Money
from src.engine.customer_id import identify_customer, normalize_name, verify_llm_customer
from src.engine.filters import filter_tx
from src.engine.invoice_match import find_invoice
from src.engine.policy import MatchPolicy
from src.engine.results import (
    CustomerOutcome,
    ExceptionReason,
    FilterOutcome,
    FilterReason,
    IdentifiedBy,
    MatchKind,
)

P = MatchPolicy()
DAY = date(2026, 11, 17)

CUSTOMERS = [
    Customer(customer_id="C001", name="שופ-נט מרקטפלייס בע״מ", currency="ILS", withholding_rate=Decimal("0")),
    Customer(customer_id="C004", name="רשת סופר-בית בע״מ", currency="ILS", withholding_rate=Decimal("0.03")),
    Customer(customer_id="C006", name="טק-גאדג'טס בע״מ", currency="ILS", withholding_rate=Decimal("0")),
    Customer(customer_id="C007", name="ספרים ועוד בע״מ", currency="ILS", withholding_rate=Decimal("0")),
    Customer(customer_id="C009", name="Pacific Trade LLC", currency="USD", withholding_rate=Decimal("0")),
    Customer(customer_id="C010", name="מטבחי השרון בע״מ", currency="ILS", withholding_rate=Decimal("0")),
    Customer(customer_id="C012", name="בטא-טק בע״מ", currency="ILS", withholding_rate=Decimal("0")),
    Customer(customer_id="C013", name="בטא שיווק בע״מ", currency="ILS", withholding_rate=Decimal("0")),
]
ALIASES = [
    Alias(bank_name="סופר בית רשתות", customer_id="C004"),
    Alias(bank_name="פייפלו תשלומים", customer_id="C006"),
    Alias(bank_name="פייפלו תשלומים", customer_id="C007"),
]


def tx(amount, text, ccy="ILS", account="ACC-01", day=DAY):
    return BankTransaction(tx_id="TX-1", tx_date=day, account_id=account, amount=Decimal(amount), currency=ccy, bank_text=text)


def inv(iid, cid, amount, ccy="ILS", issued="2026-10-01", open_balance=None):
    return OpenInvoice(invoice_id=iid, customer_id=cid, issue_date=date.fromisoformat(issued), amount=Decimal(amount),
                       currency=ccy, open_balance=Decimal(open_balance or amount))


class FakeFx:
    """שערים קבועים לבדיקה."""

    def __init__(self, rates=None, fail=False):
        self.rates, self.fail = rates or {}, fail

    def rate(self, currency, on):
        if self.fail:
            raise FxRateUnavailable("no rate")
        return FxRate(rate_date=on, currency=currency, rate_to_ils=Decimal(self.rates[on.isoformat()]))


def run(t, invoices, fx=None, paid=()):
    """מריץ זיהוי + מציאת חשבונית, כמו שהגרף יעשה."""
    cm = identify_customer(t, CUSTOMERS, ALIASES, invoices, P)
    assert cm.outcome == CustomerOutcome.IDENTIFIED, cm
    return find_invoice(t, cm, invoices, CUSTOMERS, fx or FakeFx(), P, list(paid))


# ---------- סינון ----------


def test_filters():
    assert filter_tx(tx("-500", "חברת החשמל"), P).filter_reason == FilterReason.OUTGOING
    assert filter_tx(tx("25000", "העברה בין חשבונות - מחשבון 6"), P).filter_reason == FilterReason.INTERNAL_TRANSFER
    assert filter_tx(tx("152.40", "ריבית זכות", account="ACC-06"), P).filter_reason == FilterReason.INTEREST
    assert filter_tx(tx("1863", "החזר מס - רשות המסים"), P).filter_reason == FilterReason.TAX_REFUND
    assert filter_tx(tx("9871.20", "זיכוי סליקה - כרטיסי אשראי", account="ACC-04"), P).outcome == FilterOutcome.UNSUPPORTED
    assert filter_tx(tx("4200", "שופ-נט מרקטפלייס בע״מ"), P).outcome == FilterOutcome.PASS


# ---------- זיהוי לקוח ----------


def test_normalize_name():
    assert normalize_name("שופ-נט מרקטפלייס בע״מ") == "שופנט מרקטפלייס"
    assert normalize_name("טק-גאדג׳טס") == normalize_name("טק-גאדג'טס בע״מ")  # גרש עברי = גרש לועזי
    assert normalize_name("Pacific Trade LLC") == "pacific trade"
    assert normalize_name("מטבחי השרון חש INV-10027") == "מטבחי השרון חש"


def test_identify_order():
    invoices = [inv("INV-10027", "C010", "3600")]
    assert identify_customer(tx("1800", "תשלום INV-10027"), CUSTOMERS, ALIASES, invoices, P).identified_by == IdentifiedBy.INVOICE_REF
    exact = identify_customer(tx("1", "מטבחי השרון"), CUSTOMERS, ALIASES, [], P)
    assert exact.identified_by == IdentifiedBy.EXACT_NAME and exact.strong
    alias = identify_customer(tx("1", "סופר בית רשתות"), CUSTOMERS, ALIASES, [], P)
    assert alias.identified_by == IdentifiedBy.ALIAS and not alias.strong  # כינוי שלא אושר 3 פעמים - לא ודאי


def test_close_names_go_to_llm():
    # "בטא" דומה באותה מידה לבטא-טק ולבטא שיווק - לא בוחרים, שולחים למודל
    cm = identify_customer(tx("2600", "בטא"), CUSTOMERS, ALIASES, [], P)
    assert cm.outcome == CustomerOutcome.NEEDS_LLM
    assert {c.customer_id for c in cm.candidates} == {"C012", "C013"}


def test_unknown_text_goes_to_llm_with_all_customers():
    cm = identify_customer(tx("700", "העברה מזהה 88231"), CUSTOMERS, ALIASES, [], P)
    assert cm.outcome == CustomerOutcome.NEEDS_LLM and len(cm.candidates) == len(CUSTOMERS)


def test_verify_llm_customer():
    assert verify_llm_customer("C999", CUSTOMERS).outcome == CustomerOutcome.NOT_VERIFIED
    assert verify_llm_customer(None, CUSTOMERS).outcome == CustomerOutcome.NOT_VERIFIED
    assert verify_llm_customer("C012", CUSTOMERS).identified_by == IdentifiedBy.LLM


# ---------- מציאת חשבונית ----------


def test_exact_is_certain():
    m = run(tx("4200", "שופ-נט מרקטפלייס בע״מ"), [inv("INV-10001", "C001", "4200")])
    assert m.certainty == Certainty.CERTAIN and m.kind == MatchKind.EXACT


def test_combination():
    invoices = [inv("INV-10002", "C001", "3850", issued="2026-10-12"), inv("INV-10003", "C001", "6100", issued="2026-10-19"),
                inv("INV-10004", "C001", "2950", issued="2026-10-26"), inv("INV-10005", "C001", "5400", issued="2026-11-02")]
    m = run(tx("12900", "שופ-נט מרקטפלייס בע״מ"), invoices)
    assert m.kind == MatchKind.COMBINATION
    assert [a.invoice_id for a in m.applications] == ["INV-10002", "INV-10003", "INV-10004"]


def test_two_options_same_customer_suggests_oldest():
    invoices = [inv("INV-1", "C010", "500", issued="2026-10-20"), inv("INV-2", "C010", "500", issued="2026-10-05")]
    m = run(tx("500", "מטבחי השרון"), invoices)
    assert m.applications[0].invoice_id == "INV-2"  # הוותיקה
    assert m.certainty == Certainty.PROBABLE and m.notes  # לא ודאית, ויש הערה


def test_shared_alias_amount_decides():
    invoices = [inv("INV-10019", "C006", "1890"), inv("INV-10021", "C007", "1250")]
    assert run(tx("1890", "פייפלו תשלומים"), invoices).customer_id == "C006"
    assert run(tx("1250", "פייפלו תשלומים"), invoices).customer_id == "C007"


def test_shared_alias_both_fit_is_ambiguous():
    invoices = [inv("INV-10020", "C006", "2340"), inv("INV-10022", "C007", "2340")]
    assert run(tx("2340", "פייפלו תשלומים"), invoices).exception_reason == ExceptionReason.AMBIGUOUS_MATCH


def test_bank_fee_closes_invoice():
    m = run(tx("2980", "מטבחי השרון"), [inv("INV-10032", "C010", "3000")])
    assert m.kind == MatchKind.BANK_FEE
    assert m.receipt_amount == Money.of("3000", "ILS")  # החשבונית נסגרת על הסכום המלא
    assert m.adjustments[0].amount == Money.of("20", "ILS")


def test_fee_above_limit_is_not_fee():
    m = run(tx("2970", "מטבחי השרון"), [inv("INV-10032", "C010", "3000")])  # 30 > 25
    assert m.is_exception


def test_foreign_fee_limit():
    m = run(tx("4462", "Pacific Trade LLC", ccy="USD", account="ACC-03"), [inv("INV-10025", "C009", "4500", ccy="USD")])
    assert m.kind == MatchKind.BANK_FEE and m.adjustments[0].amount == Money.of("38", "USD")


def test_withholding_with_rounding():
    m = run(tx("9700.40", "סופר בית רשתות"), [inv("INV-10020", "C004", "10000")])
    types = {a.type: a.amount for a in m.adjustments}
    assert types[AdjustmentType.WITHHOLDING_TAX] == Money.of("300", "ILS")
    assert types[AdjustmentType.ROUNDING] == Money.of("-0.40", "ILS")


def test_fee_and_withholding_together_is_exception():
    m = run(tx("9690", "סופר בית רשתות"), [inv("INV-10020", "C004", "10000")])  # 3% וגם עמלה של 10
    assert m.is_exception


def test_fx_within_tolerance():
    rates = {"2026-11-09": "3.171", "2026-11-15": "3.199"}
    m = run(tx("7037.80", "Pacific Trade LLC", day=date(2026, 11, 15)),
            [inv("INV-10026", "C009", "2200", ccy="USD", issued="2026-11-09")], FakeFx(rates))
    assert m.kind == MatchKind.FX and m.certainty == Certainty.PROBABLE
    assert m.receipt_amount == Money.of("2200", "USD")
    assert m.adjustments[0].amount == Money.of("61.60", "ILS")


def test_fx_above_tolerance():
    rates = {"2026-09-24": "3.047", "2026-11-17": "3.246"}
    m = run(tx("3246", "Pacific Trade LLC"), [inv("INV-10034", "C009", "1000", ccy="USD", issued="2026-09-24")], FakeFx(rates))
    assert m.exception_reason == ExceptionReason.FX_DIFF_ABOVE_TOLERANCE


def test_fx_rate_unavailable():
    m = run(tx("3246", "Pacific Trade LLC"), [inv("INV-10034", "C009", "1000", ccy="USD")], FakeFx(fail=True))
    assert m.exception_reason == ExceptionReason.FX_RATE_UNAVAILABLE


def test_partial_with_reference():
    m = run(tx("1800", "מטבחי השרון INV-10027"), [inv("INV-10027", "C010", "3600")])
    assert m.kind == MatchKind.PARTIAL_REF and m.applications[0].amount == Money.of("1800", "ILS")


def test_partial_without_reference():
    m = run(tx("900", "מטבחי השרון"), [inv("INV-10027", "C010", "3600", open_balance="1800")])
    assert m.exception_reason == ExceptionReason.PARTIAL_WITHOUT_REFERENCE


def test_overpayment():
    m = run(tx("9999", "מטבחי השרון"), [inv("INV-10027", "C010", "1800")])
    assert m.exception_reason == ExceptionReason.OVERPAYMENT


def test_duplicate_payment_is_overpayment():
    paid = [Invoice(invoice_id="INV-10001", customer_id="C001", issue_date=date(2026, 10, 5), amount=Decimal("4200"), currency="ILS")]
    m = run(tx("4200", "שופ-נט מרקטפלייס בע״מ"), [inv("INV-10056", "C001", "4480")], paid=paid)
    assert m.exception_reason == ExceptionReason.OVERPAYMENT
    assert m.evidence["possible_duplicate_of"] == "INV-10001"


def test_llm_customer_without_invoice_is_not_verified():
    cm = verify_llm_customer("C001", CUSTOMERS)
    m = find_invoice(tx("2215", "ש.נ. מרקטפלייס"), cm, [inv("INV-10056", "C001", "4480")], CUSTOMERS, FakeFx(), P)
    assert m.exception_reason == ExceptionReason.LLM_SUGGESTION_NOT_VERIFIED
