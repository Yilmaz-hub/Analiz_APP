"""Spec 0003 QA düzeltmelerinin uygulama düzeyi testleri.

Gerçek `app.py`, `streamlit.testing.v1.AppTest` ile sürülür. Her test, QA
raporundaki ilgili bulgunun (Q-numarası) düzeltme geri alındığında düşmesi için
yazılmıştır.

Bulgu eşlemesi:
  Q9  -> test_every_portfolio_write_goes_through_safe_wrapper,
         test_stop_raise_failed_write_is_rolled_back_and_does_not_crash,
         test_stop_raise_success_survives_a_later_failed_write,
         test_stop_raise_button_is_disabled_when_storage_is_unwritable
"""
import ast
import os

from app_helpers import (
    APP_PATH, break_storage, break_writes_only, button, click, make_app, texts,
)

VARLIKLAR = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"}

STOP_POZISYONU = {
    "Coin": "Bitcoin (BTC)", "Giriş": 100.0, "Adet": 2.0, "Yatırım": 200.0,
    "Realized": 0.0, "Status": "ACTIVE", "Stop": 90.0, "V1Verified": True,
    "Gerçekleşme Zamanı": "2026-01-02T10:00:00+00:00", "Tarih": "2026-01-02",
}


def _portfolio():
    return {"balance": 500.0, "positions": [dict(STOP_POZISYONU)]}


def _seed(store):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, _portfolio())


def _raise_stop_to(at, value):
    at.number_input(key="raised_stop").set_value(value)


# Q9 — Portföyü yazan hiçbir yol güvenli sarmalayıcıyı atlayamaz.
def test_every_portfolio_write_goes_through_safe_wrapper():
    source = open(APP_PATH, encoding="utf-8").read()
    tree = ast.parse(source)

    wrapper_calls = {
        id(node)
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == "safe_save_portfolio"
        for node in ast.walk(fn)
    }
    bypasses = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "save_portfolio" and id(node) not in wrapper_calls
    ]

    assert bypasses == [], f"save_portfolio güvenli sarmalayıcıyı atlıyor: satır {bypasses}"


# Q9 — Yazma koptuğunda stop yükseltmesi geri alınır, kullanıcıya teknik hata düşmez.
def test_stop_raise_failed_write_is_rolled_back_and_does_not_crash(store, monkeypatch, processed_df):
    _seed(store)
    at = make_app(monkeypatch, processed_df).run()
    assert not at.exception

    geri_al = break_writes_only(store)
    try:
        _raise_stop_to(at, 95.0)
        click(at, "Stop Yükseltmesini Teyit Et")
    finally:
        geri_al()

    assert not at.exception
    assert "kaydedilmedi" in texts(at)
    bellek = at.session_state["portfolio_data"]["positions"][0]
    assert bellek["Stop"] == 90.0
    assert "Stop Geçmişi" not in bellek
    assert store.read_doc(store.PORTFOLIO_KEY)["positions"][0]["Stop"] == 90.0


# Q9 — Başarılı yükseltme geri alma noktasını tazeler; sonraki bir başarısız
# yazma onu silmez.
def test_stop_raise_success_survives_a_later_failed_write(store, monkeypatch, processed_df):
    _seed(store)
    at = make_app(monkeypatch, processed_df).run()

    _raise_stop_to(at, 95.0)
    click(at, "Stop Yükseltmesini Teyit Et")
    assert store.read_doc(store.PORTFOLIO_KEY)["positions"][0]["Stop"] == 95.0

    # Başka bir işlem yazma kopukken denenir ve geri alınır.
    geri_al = break_writes_only(store)
    try:
        bakiye = next(n for n in at.number_input if n.label == "Güncel USDT Bakiyesi")
        bakiye.set_value(999.0)
        click(at, "Bakiyeyi Güncelle")
    finally:
        geri_al()

    assert at.session_state["portfolio_data"]["positions"][0]["Stop"] == 95.0

    # Erişim düzeldi, başka işlem yapıldı: yükseltilen stop depoda kalmalı.
    at.run()
    bakiye = next(n for n in at.number_input if n.label == "Güncel USDT Bakiyesi")
    bakiye.set_value(1234.0)
    click(at, "Bakiyeyi Güncelle")

    kayitli = store.read_doc(store.PORTFOLIO_KEY)
    assert kayitli["balance"] == 1234.0
    assert kayitli["positions"][0]["Stop"] == 95.0


# Q9 — Erişim oturum ortasında koparsa stop yükseltme düğmesi kapanır.
def test_stop_raise_button_is_disabled_when_storage_is_unwritable(store, monkeypatch, processed_df):
    _seed(store)
    at = make_app(monkeypatch, processed_df).run()
    assert button(at, "Stop Yükseltmesini Teyit Et").disabled is False

    geri_al = break_storage(store)
    try:
        at.run()
        assert not at.exception
        assert button(at, "Stop Yükseltmesini Teyit Et").disabled is True
    finally:
        geri_al()


# Q6 — Hesaplanamayan karar bileşeni nötr sayılmaz; ekranda adıyla görünür.
def test_q6_sidebar_names_the_unavailable_component(store, monkeypatch, processed_df):
    import ml_models
    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", lambda _df: None)
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)

    at = make_app(monkeypatch, _taze_gunluk(processed_df)).run()

    assert not at.exception
    metin = texts(at)
    assert "Karar bileşeni hazır değil" in metin and "yapay zekâ" in metin
    assert "BILESEN_HAZIR_DEGIL" not in metin


def _taze_gunluk(frame):
    """Son kapanmış gün dünkü olacak biçimde kaydırılmış (geçerli) günlük veri."""
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    frame = frame.copy()
    frame["Open"] = frame["Open"].clip(lower=frame["Low"], upper=frame["High"])  # geçerli OHLC
    dun = datetime.now(timezone.utc).date() - timedelta(days=1)
    frame.index = pd.date_range(end=pd.Timestamp(dun), periods=len(frame), freq="D", tz="UTC")
    return frame
