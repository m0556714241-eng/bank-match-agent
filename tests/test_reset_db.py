"""בדיקות לאיפוס ולהגדרות."""

from datetime import date

import pytest

import sys

from scripts import reset_db
from scripts.reset_db import SeedFileError, reset
from src.config import load_settings

CUSTOMERS = "customer_id,name,currency,withholding_rate\nC010,מטבחי השרון,ILS,0.0\n"
INVOICES = (
    "invoice_id,customer_id,date,amount,currency,open_balance\n"
    "INV-10042,C010,2026-09-17,2200.00,ILS,1100.00\n"
)
ALIASES = "bank_name,customer_id,times_confirmed\nמטבחי השרון,C010,0\n"


def _files(tmp_path, invoices=INVOICES):
    paths = {}
    for name, text in [("customers", CUSTOMERS), ("invoices", invoices), ("aliases", ALIASES)]:
        p = tmp_path / f"{name}.csv"
        p.write_text(text, encoding="utf-8-sig")  # עם BOM, כמו Excel
        paths[name] = p
    return paths


def test_reset_builds_db(tmp_path):
    f = _files(tmp_path)
    ledger = reset(tmp_path / "out" / "ledger.db", f["customers"], f["invoices"], f["aliases"])
    [inv] = ledger.open_invoices(date(2026, 11, 30))
    assert str(inv.open_balance) == "1100.00"


def test_reset_twice_starts_fresh(tmp_path):
    f = _files(tmp_path)
    db = tmp_path / "ledger.db"
    reset(db, f["customers"], f["invoices"], f["aliases"])
    ledger = reset(db, f["customers"], f["invoices"], f["aliases"])  # שוב - בלי שגיאה
    assert len(ledger.customers()) == 1


def test_bad_file_keeps_old_db(tmp_path):
    f = _files(tmp_path)
    db = tmp_path / "ledger.db"
    reset(db, f["customers"], f["invoices"], f["aliases"])
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_invoices = "invoice_id,customer_id,date,amount,currency,open_balance\nINV-1,C010,xx,5,ILS,5\n"
    bad = _files(bad_dir, invoices=bad_invoices)
    with pytest.raises(SeedFileError):
        reset(db, bad["customers"], bad["invoices"], bad["aliases"])
    assert db.exists()  # המסד הישן לא נמחק


def test_settings_paths_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    s = load_settings(env)
    assert s.db_path.name == "ledger.db"
    assert s.anthropic_api_key is None  # שדה ריק = אין מפתח


def test_reset_blocked_in_production(monkeypatch, capsys):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(sys, "argv", ["reset_db", "--yes"])
    assert reset_db.main() == 1
    assert "חסום" in capsys.readouterr().out


def test_invalid_app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "staging")
    with pytest.raises(ValueError):
        load_settings(tmp_path / "none.env")
