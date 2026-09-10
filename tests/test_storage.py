"""Kayıt deposu testleri — spec 0004.

Kriter eşlemesi:
  AC01  -> test_prepared_records_load_unchanged
  AC01b -> test_migration_copies_legacy_records_verbatim,
           test_migration_does_not_touch_source_file,
           test_migration_does_not_overwrite_existing_records
  AC05a -> test_records_survive_process_restart
  AC06  -> test_thirty_day_old_records_load_unchanged
  AC16  -> test_read_error_raises_instead_of_returning_defaults,
           test_records_return_unchanged_after_access_is_restored
  AC18  -> test_records_independent_of_working_directory
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

import portfolio as portfolio_module
import storage

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HAZIR_VARLIKLAR = {"Bitcoin (BTC)": "BTC-USD"}
HAZIR_PORTFOY = {
    "balance": 2500.0,
    "positions": [
        {"Coin": "Bitcoin (BTC)", "Giriş": 42000.0, "Adet": 0.25,
         "Yatırım": 10500.0, "Realized": 0.0, "Status": "ACTIVE",
         "Tarih": "2025-12-01"}
    ],
}


# AC01 — Geliştirme öncesinde hazırlanmış kayıtlar bilgileriyle bulunur.
def test_prepared_records_load_unchanged(store):
    store.write_doc(store.ASSETS_KEY, HAZIR_VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, HAZIR_PORTFOY)

    store.reset_engine()  # yeniden açılışı taklit eder

    assert store.read_doc(store.ASSETS_KEY) == HAZIR_VARLIKLAR
    assert store.read_doc(store.PORTFOLIO_KEY) == HAZIR_PORTFOY


# AC01b — Eski dosya kayıtları birebir içeri alınır.
def test_migration_copies_legacy_records_verbatim(store, legacy_dir):
    (legacy_dir / "portfolio.json").write_text(
        json.dumps(HAZIR_PORTFOY), encoding="utf-8")
    (legacy_dir / "varliklar.json").write_text(
        json.dumps(HAZIR_VARLIKLAR, ensure_ascii=False), encoding="utf-8")

    imported = store.import_legacy_documents()

    assert set(imported) == {store.ASSETS_KEY, store.PORTFOLIO_KEY}
    assert store.read_doc(store.PORTFOLIO_KEY) == HAZIR_PORTFOY
    assert store.read_doc(store.ASSETS_KEY) == HAZIR_VARLIKLAR


# AC01b — Kaynak dosya korunur; referans kopyası taşınmaz ya da silinmez.
def test_migration_does_not_touch_source_file(store, legacy_dir):
    source = legacy_dir / "portfolio.json"
    source.write_text(json.dumps(HAZIR_PORTFOY), encoding="utf-8")
    before = source.read_bytes()

    store.import_legacy_documents()

    assert source.is_file()
    assert source.read_bytes() == before


# AC01b — Depoda kayıt varken eski dosya onun üzerine yazmaz.
def test_migration_does_not_overwrite_existing_records(store, legacy_dir):
    store.write_doc(store.PORTFOLIO_KEY, HAZIR_PORTFOY)
    (legacy_dir / "portfolio.json").write_text(
        json.dumps({"balance": 1.0, "positions": []}), encoding="utf-8")

    imported = store.import_legacy_documents()

    assert store.PORTFOLIO_KEY not in imported
    assert store.read_doc(store.PORTFOLIO_KEY) == HAZIR_PORTFOY


# AC05a — Süreç tamamen sonlandırılıp yeniden başlatıldığında kayıtlar aynıdır.
def test_records_survive_process_restart(store, tmp_path, monkeypatch):
    store.write_doc(store.ASSETS_KEY, HAZIR_VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, HAZIR_PORTFOY)

    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, sys.argv[1])
        import storage
        print(json.dumps({
            "assets": storage.read_doc(storage.ASSETS_KEY),
            "portfolio": storage.read_doc(storage.PORTFOLIO_KEY),
        }))
        """
    )
    script_path = tmp_path / "read_records.py"
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    env[storage.DATA_DIR_ENV] = os.environ[storage.DATA_DIR_ENV]
    out = subprocess.run(
        [sys.executable, str(script_path), PROJECT_DIR],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120,
    )
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout.strip().splitlines()[-1])

    assert payload["assets"] == HAZIR_VARLIKLAR
    assert payload["portfolio"] == HAZIR_PORTFOY


# AC18 — Kayıtlar, uygulamanın başlatıldığı klasörden bağımsızdır.
def test_records_independent_of_working_directory(store, tmp_path):
    store.write_doc(store.ASSETS_KEY, HAZIR_VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, HAZIR_PORTFOY)

    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, sys.argv[1])
        import storage
        print(json.dumps(storage.read_doc(storage.PORTFOLIO_KEY)))
        """
    )
    script_path = tmp_path / "read_from_anywhere.py"
    script_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    results = []
    for folder in ("klasor_a", "klasor_b"):
        work_dir = tmp_path / folder
        work_dir.mkdir()
        out = subprocess.run(
            [sys.executable, str(script_path), PROJECT_DIR],
            capture_output=True, text=True, env=env, cwd=str(work_dir), timeout=120,
        )
        assert out.returncode == 0, out.stderr
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))

    assert results[0] == results[1] == HAZIR_PORTFOY


# AC06 — 30 gün önceye ayarlanmış kayıtlar bilgileriyle bulunur.
def test_thirty_day_old_records_load_unchanged(store):
    eski_portfoy = json.loads(json.dumps(HAZIR_PORTFOY))
    eski_portfoy["positions"][0]["Tarih"] = "2025-11-01"
    store.write_doc(store.ASSETS_KEY, HAZIR_VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, eski_portfoy)

    # Kayıt dosyasının yaşı 30 gün geriye alınır.
    db_file = store.data_dir() / "analiz.db"
    store.reset_engine()
    thirty_days = 30 * 24 * 3600
    old_time = os.path.getmtime(db_file) - thirty_days
    os.utime(db_file, (old_time, old_time))

    assert store.read_doc(store.PORTFOLIO_KEY) == eski_portfoy
    assert store.read_doc(store.ASSETS_KEY) == HAZIR_VARLIKLAR


# AC16 — Erişim sorununda varsayılan/boş veri döndürülmez.
def test_read_error_raises_instead_of_returning_defaults(store, monkeypatch):
    store.write_doc(store.PORTFOLIO_KEY, HAZIR_PORTFOY)

    def kirik_engine():
        raise store.StorageAccessError("test")

    monkeypatch.setattr(store, "get_engine", kirik_engine)

    with pytest.raises(store.StorageAccessError):
        store.read_doc(store.PORTFOLIO_KEY)
    with pytest.raises(store.StorageAccessError):
        portfolio_module.load_portfolio()
    assert store.check_access()[0] is False


# AC16 — Erişim düzeldiğinde kayıtlar önceki bilgileriyle bulunur.
def test_records_return_unchanged_after_access_is_restored(store, monkeypatch):
    store.write_doc(store.PORTFOLIO_KEY, HAZIR_PORTFOY)
    gercek_engine = store.get_engine

    monkeypatch.setattr(store, "get_engine",
                        lambda: (_ for _ in ()).throw(store.StorageAccessError("test")))
    with pytest.raises(store.StorageAccessError):
        portfolio_module.load_portfolio()

    monkeypatch.setattr(store, "get_engine", gercek_engine)

    assert portfolio_module.load_portfolio() == HAZIR_PORTFOY
    assert store.check_access()[0] is True
