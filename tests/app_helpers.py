"""AppTest tabanlı testlerin ortak yardımcıları.

`tests/test_persistence_app.py` (spec 0004) ve spec 0003'ün QA düzeltme testleri
aynı uygulama sürücüsünü kullanır; ortak parça burada tek yerde durur.
"""
import os

from streamlit.testing.v1 import AppTest

import data_fetchers

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

#: Depoya erişimin geçtiği tüm giriş noktaları — biri açık bırakılırsa
#: "erişim koptu" senaryosu eksik taklit edilir.
DEPO_GIRISLERI = ("read_doc", "write_doc", "ping", "get_engine")


def make_app(monkeypatch, frame, *, live_price=45000.0):
    """Ağ kaynakları taklit edilmiş bir AppTest üretir.

    app.py `from data_fetchers import ...` ile bağladığı için kaynak modülün
    özniteliklerini değiştirmek yeterlidir.
    """
    monkeypatch.setattr(data_fetchers, "get_market_data",
                        lambda *a, **k: (frame, "Binance"))
    monkeypatch.setattr(data_fetchers, "get_fear_greed_index", lambda: (50, "Neutral"))
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        lambda *a, **k: live_price)
    return AppTest.from_file(APP_PATH, default_timeout=180)


def texts(at):
    """Ekrandaki tüm bilgi/uyarı/hata metinleri."""
    parts = []
    for kind in (at.info, at.warning, at.error, at.success, at.caption, at.markdown):
        parts.extend(str(getattr(e, "value", "")) for e in kind)
    return " ".join(parts)


def instrument_options(at):
    for sb in at.selectbox:
        if sb.label == "Enstrüman:":
            return list(sb.options)
    return []


def click(at, label):
    """Etiketine göre bir butona basar ve uygulamayı yeniden çalıştırır."""
    for button in at.button:
        if button.label == label:
            button.click().run()
            return at
    raise AssertionError(f"'{label}' butonu bulunamadı")


def button(at, label):
    """Etiketine göre butonu döndürür (basmadan)."""
    for candidate in at.button:
        if candidate.label == label:
            return candidate
    raise AssertionError(f"'{label}' butonu bulunamadı")


def break_storage(store):
    """Depoyu erişilemez yapar; geri alma işlevini döndürür.

    monkeypatch yerine elle geri alınır: fixture'ın kendi yamaları (veri
    klasörü) testin ortasında geri alınmamalıdır.
    """
    gercek = {ad: getattr(store, ad) for ad in DEPO_GIRISLERI}

    def kirik(*a, **k):
        raise store.StorageAccessError("test")

    for ad in DEPO_GIRISLERI:
        setattr(store, ad, kirik)

    def geri_al():
        for ad, fn in gercek.items():
            setattr(store, ad, fn)

    return geri_al


def break_writes_only(store):
    """Yalnız yazmayı bozar; okuma çalışmaya devam eder.

    Erişim oturumun ortasında kopan, kontrollerin hâlâ çizildiği durumu
    taklit eder (QA F1'in reprosu).
    """
    gercek_write = store.write_doc

    def kirik(*a, **k):
        raise store.StorageAccessError("test")

    store.write_doc = kirik

    def geri_al():
        store.write_doc = gercek_write

    return geri_al
