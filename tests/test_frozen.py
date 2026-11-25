"""בדיקת העזר freeze_sequences (ה-71): רשימה שהגיעה הופכת ל-tuple."""

from src.adapters.interfaces import Certainty
from src.engine.results import InvoiceMatch


def test_lists_become_tuples():
    match = InvoiceMatch(certainty=Certainty.PROBABLE, notes=["א"], applications=[])  # כמו אחרי טעינה מהקובץ
    assert match.notes == ("א",)
    assert isinstance(match.applications, tuple)
