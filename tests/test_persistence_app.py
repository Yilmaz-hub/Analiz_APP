"""Uygulama düzeyi koruma testleri — spec 0004.

Gerçek `app.py`, `streamlit.testing.v1.AppTest` ile çalıştırılır; ağ kaynakları
taklit edilir. Yalnız uygulama katmanında görülebilen kriterler buradadır.

Kriter eşlemesi:
  AC02  -> test_added_asset_is_found_after_reopen
  AC03  -> test_page_refresh_keeps_records_unchanged
  AC04  -> test_reopening_app_keeps_records_unchanged
  AC07  -> test_empty_state_shows_message_and_creates_nothing
  AC12  -> test_delete_attempt_on_asset_with_active_position_is_refused
  AC13  -> test_switching_instrument_keeps_both_assets
  AC14  -> test_only_active_position_is_listed_but_asset_remains
  AC15b -> test_duplicate_name_is_refused_in_ui
  AC16  -> test_access_problem_is_shown_to_user
  AC16b -> test_no_write_controls_while_access_is_broken,
           test_write_controls_are_disabled_when_access_breaks_mid_session (QA F2),
           test_failed_write_is_rolled_back_in_memory (QA F1),
           test_failed_write_does_not_become_permanent_later (QA F1)
  R4    -> test_stop_alert_does_not_change_records (QA F10 / spec 0003)
  AC16c -> test_broken_record_is_shown_and_kept
  R8.2  -> test_unknown_asset_does_not_show_another_assets_data
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

import data_fetchers

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

VARLIKLAR = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"}
AKTIF_POZISYON = {
    "Coin": "Bitcoin (BTC)", "Giriş": 42000.0, "Adet": 0.25, "Yatırım": 10500.0,
    "Realized": 0.0, "Status": "ACTIVE", "Tarih": "2025-12-01",
}
PORTFOY = {"balance": 2500.0, "positions": [AKTIF_POZISYON]}


def _make_app(monkeypatch, processed_df):
    """Ağ kaynakları taklit edilmiş bir AppTest üretir.

    app.py `from data_fetchers import ...` ile bağladığı için kaynak modülün
    özniteliklerini değiştirmek yeterlidir.
    """
    monkeypatch.setattr(data_fetchers, "get_market_data",
                        lambda *a, **k: (processed_df, "Binance"))
    monkeypatch.setattr(data_fetchers, "get_fear_greed_index", lambda: (50, "Neutral"))
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        lambda *a, **k: 45000.0)
    return AppTest.from_file(APP_PATH, default_timeout=180)


def _texts(at):
    """Ekrandaki tüm bilgi/uyarı/hata metinleri."""
    parts = []
    for kind in (at.info, at.warning, at.error, at.success, at.caption, at.markdown):
        parts.extend(str(getattr(e, "value", "")) for e in kind)
    return " ".join(parts)


def _instrument_options(at):
    for sb in at.selectbox:
        if sb.label == "Enstrüman:":
            return list(sb.options)
    return []


def _stored(store):
    return store.read_doc(store.ASSETS_KEY), store.read_doc(store.PORTFOLIO_KEY)


# AC07 — Kayıt yokken boş durum gösterilir, kendiliğinden kayıt oluşmaz.
def test_empty_state_shows_message_and_creates_nothing(store, monkeypatch, processed_df):
    at = _make_app(monkeypatch, processed_df).run()

    assert not at.exception
    assert "Henüz kayıtlı varlık yok" in _texts(at)
    # Hiçbir varlık adı görünmez.
    assert _instrument_options(at) == []
    for ad in ("Bitcoin", "Ethereum", "Pepe"):
        assert ad not in _texts(at)
    # Depoda kayıt oluşmamıştır.
    assert _stored(store) == (None, None)


# AC04 — Uygulama yeniden açıldığında kayıtlar bilgileriyle aynıdır.
def test_reopening_app_keeps_records_unchanged(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, PORTFOY)

    at = _make_app(monkeypatch, processed_df).run()
    assert not at.exception
    assert set(_instrument_options(at)) == set(VARLIKLAR)

    # "Kapat ve yeniden aç": yeni bir uygulama örneği.
    store.reset_engine()
    at2 = _make_app(monkeypatch, processed_df).run()

    assert not at2.exception
    assert set(_instrument_options(at2)) == set(VARLIKLAR)
    assert _stored(store) == (VARLIKLAR, PORTFOY)


# AC03 — Sayfa yenilendiğinde kayıtların bilgileri değişmez.
def test_page_refresh_keeps_records_unchanged(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, PORTFOY)

    at = _make_app(monkeypatch, processed_df).run()
    at.run()  # yenileme

    assert not at.exception
    assert set(_instrument_options(at)) == set(VARLIKLAR)
    assert _stored(store) == (VARLIKLAR, PORTFOY)


# AC02 — Eklenen pozisyonsuz varlık yeniden açılışta bulunur.
def test_added_asset_is_found_after_reopen(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)

    at = _make_app(monkeypatch, processed_df).run()
    inputs = [t for t in at.text_input if t.label.startswith(("Görünen", "Yahoo"))]
    assert len(inputs) == 2
    inputs[0].set_value("Pound")
    inputs[1].set_value("GBPUSD=X")
    _click(at, "Listeye Ekle")

    kayitli = store.read_doc(store.ASSETS_KEY)
    assert kayitli["Pound"] == "GBPUSD=X"

    store.reset_engine()
    at2 = _make_app(monkeypatch, processed_df).run()
    assert "Pound" in _instrument_options(at2)


def _click(at, label):
    """Etiketine göre bir butona basar ve uygulamayı yeniden çalıştırır."""
    for button in at.button:
        if button.label == label:
            button.click().run()
            return at
    raise AssertionError(f"'{label}' butonu bulunamadı")


# AC15b — Kayıtlı adla ekleme reddedilir, mevcut kayıt değişmez.
def test_duplicate_name_is_refused_in_ui(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)

    at = _make_app(monkeypatch, processed_df).run()
    inputs = [t for t in at.text_input if t.label.startswith(("Görünen", "Yahoo"))]
    inputs[0].set_value("Bitcoin (BTC)")
    inputs[1].set_value("BASKA-KOD")
    _click(at, "Listeye Ekle")

    assert "zaten kayıtlı" in _texts(at)
    assert store.read_doc(store.ASSETS_KEY) == VARLIKLAR


# AC12 — Aktif pozisyonlu varlık silinemez; kayıtlar korunur.
def test_delete_attempt_on_asset_with_active_position_is_refused(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, PORTFOY)

    at = _make_app(monkeypatch, processed_df).run()
    at.selectbox(key="del_box").set_value("Bitcoin (BTC)")
    _click(at, "Seçileni Sil")

    assert "aktif pozisyon" in _texts(at)
    assert _stored(store) == (VARLIKLAR, PORTFOY)


# AC13 — Enstrüman değiştirip geri dönmek kayıtları etkilemez.
def test_switching_instrument_keeps_both_assets(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, PORTFOY)

    at = _make_app(monkeypatch, processed_df).run()
    enstruman = next(sb for sb in at.selectbox if sb.label == "Enstrüman:")
    enstruman.set_value("Ethereum (ETH)").run()
    assert set(_instrument_options(at)) == set(VARLIKLAR)

    enstruman = next(sb for sb in at.selectbox if sb.label == "Enstrüman:")
    enstruman.set_value("Bitcoin (BTC)").run()

    assert set(_instrument_options(at)) == set(VARLIKLAR)
    assert _stored(store) == (VARLIKLAR, PORTFOY)


# AC14 — Aktif Pozisyonlar'da yalnız aktif olan yer alır; kapalının varlığı kalır.
def test_only_active_position_is_listed_but_asset_remains(store, monkeypatch, processed_df):
    kapali = dict(AKTIF_POZISYON, Coin="Ethereum (ETH)", Adet=0.0)
    portfoy = {"balance": 2500.0, "positions": [AKTIF_POZISYON, kapali]}
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, portfoy)

    at = _make_app(monkeypatch, processed_df).run()

    assert not at.exception
    aktif_tablolar = [df.value for df in at.dataframe
                      if "Kar/Zarar (%)" in getattr(df.value, "columns", [])]
    assert aktif_tablolar, "Aktif pozisyon tablosu bulunamadı"
    listelenen = set(aktif_tablolar[0]["Coin"])
    assert listelenen == {"Bitcoin (BTC)"}
    # Kapalı pozisyonun varlığı listede kalır.
    assert "Ethereum (ETH)" in _instrument_options(at)


# AC16c — Bilgileri eksik kayıt korunur ve sorunlu olarak gösterilir.
def test_broken_record_is_shown_and_kept(store, monkeypatch, processed_df):
    bozuk = dict(AKTIF_POZISYON, Coin="Ethereum (ETH)", Adet="okunamaz")
    portfoy = {"balance": 2500.0, "positions": [AKTIF_POZISYON, bozuk]}
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, portfoy)

    at = _make_app(monkeypatch, processed_df).run()

    assert not at.exception
    assert "Sorunlu Kayıtlar" in _texts(at)
    assert store.read_doc(store.PORTFOLIO_KEY) == portfoy


# R8.2 — Bilinmeyen varlık için başka bir varlığın bilgileri gösterilmez.
def test_unknown_asset_does_not_show_another_assets_data(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    at = _make_app(monkeypatch, processed_df).run()

    # Seçim yapıldıktan sonra varlık kayıtlardan kalkarsa BTC verisine düşülmez.
    store.write_doc(store.ASSETS_KEY, {"Ethereum (ETH)": "ETH-USD"})
    at.session_state["coin_map"] = {"Ethereum (ETH)": "ETH-USD"}
    at.run()

    metin = _texts(at)
    assert "Bitcoin (BTC)" not in _instrument_options(at)
    assert "bulunamadı" in metin or "Ethereum" in str(_instrument_options(at))


#: Depoya erişimin geçtiği tüm giriş noktaları — biri açık bırakılırsa
#: "erişim koptu" senaryosu eksik taklit edilir.
_DEPO_GIRISLERI = ("read_doc", "write_doc", "ping", "get_engine")


def _break_storage(store):
    """Depoyu erişilemez yapar; geri alma işlevini döndürür.

    monkeypatch yerine elle geri alınır: fixture'ın kendi yamaları (veri
    klasörü) testin ortasında geri alınmamalıdır.
    """
    gercek = {ad: getattr(store, ad) for ad in _DEPO_GIRISLERI}

    def kirik(*a, **k):
        raise store.StorageAccessError("test")

    for ad in _DEPO_GIRISLERI:
        setattr(store, ad, kirik)

    def geri_al():
        for ad, fn in gercek.items():
            setattr(store, ad, fn)

    return geri_al


# AC16 — Erişim sorunu kullanıcıya gösterilir; kayıtlar boşaltılmaz.
def test_access_problem_is_shown_to_user(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, PORTFOY)

    at = _make_app(monkeypatch, processed_df)
    geri_al = _break_storage(store)
    try:
        at.run()

        assert not at.exception
        metin = _texts(at)
        assert "erişilemiyor" in metin
        assert "silinmedi" in metin
        # Teknik hata metni sızmaz.
        assert "Traceback" not in metin and "StorageAccessError" not in metin
    finally:
        geri_al()

    # Erişim düzeldiğinde kayıtlar önceki bilgileriyle bulunur.
    store.reset_engine()
    at2 = _make_app(monkeypatch, processed_df).run()
    assert set(_instrument_options(at2)) == set(VARLIKLAR)
    assert _stored(store) == (VARLIKLAR, PORTFOY)


# AC16b — Erişim sorunu sürerken kayıt değiştiren kontrol sunulmaz.
def test_no_write_controls_while_access_is_broken(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, PORTFOY)
    onceki = _stored(store)

    at = _make_app(monkeypatch, processed_df)
    geri_al = _break_storage(store)
    try:
        at.run()

        etiketler = [b.label for b in at.button]
        for yasak in ("Listeye Ekle", "Seçileni Sil", "🔄 Varsayılan Listeyi Yükle",
                      "🗑️ Portföyü Sıfırla", "➕ Emri Gir / Ekle"):
            assert yasak not in etiketler
    finally:
        geri_al()

    store.reset_engine()
    assert _stored(store) == onceki


BEKLEYEN_EMIR = {
    "Coin": "Bitcoin (BTC)", "Giriş": 40000.0, "Adet": 0.00125, "Yatırım": 50.0,
    "Realized": 0.0, "Status": "PENDING", "Tarih": "2025-12-01",
}
EMIRLI_PORTFOY = {"balance": 110.0, "positions": [BEKLEYEN_EMIR]}


def _break_writes_only(store):
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


# AC16b / QA F1 — Yazma koptuğunda bellekteki değişiklik geri alınır.
def test_failed_write_is_rolled_back_in_memory(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, EMIRLI_PORTFOY)

    at = _make_app(monkeypatch, processed_df).run()
    assert not at.exception

    geri_al = _break_writes_only(store)
    try:
        _click(at, "❌ İptal Et")
    finally:
        geri_al()

    # Kullanıcıya işlemin kaydedilmediği bildirilir.
    assert "kaydedilmedi" in _texts(at)
    # Depo değişmemiştir.
    assert store.read_doc(store.PORTFOLIO_KEY) == EMIRLI_PORTFOY
    # Bellek de geri alınmıştır: emir duruyor, bakiye artmamış.
    bellek = at.session_state["portfolio_data"]
    assert len(bellek["positions"]) == 1
    assert bellek["positions"][0]["Status"] == "PENDING"
    assert bellek["balance"] == 110.0


# AC16b / QA F1 — Başarısız işlem sonraki başarılı yazmayla kalıcı olmaz.
def test_failed_write_does_not_become_permanent_later(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, EMIRLI_PORTFOY)

    at = _make_app(monkeypatch, processed_df).run()
    geri_al = _break_writes_only(store)
    try:
        _click(at, "❌ İptal Et")
    finally:
        geri_al()

    # Erişim düzeldi; kullanıcı BAŞKA bir işlem yapıyor.
    at.run()
    bakiye_girisi = next(n for n in at.number_input
                         if n.label == "Güncel USDT Bakiyesi")
    bakiye_girisi.set_value(300.0)
    _click(at, "Bakiyeyi Güncelle")

    kayitli = store.read_doc(store.PORTFOLIO_KEY)
    assert kayitli["balance"] == 300.0
    # İptal edilmemiş emir hâlâ yerinde: iptal kalıcılaşmadı.
    assert len(kayitli["positions"]) == 1
    assert kayitli["positions"][0]["Status"] == "PENDING"


# AC16b / QA F2 — Erişim oturum ortasında koparsa yazan kontroller kapanır.
def test_write_controls_are_disabled_when_access_breaks_mid_session(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    store.write_doc(store.PORTFOLIO_KEY, EMIRLI_PORTFOY)

    # Önce erişim çalışırken açılır: sayfa tam çizilir, kontroller etkindir.
    at = _make_app(monkeypatch, processed_df).run()
    assert not at.exception
    acikken = {b.label: b for b in at.button}
    assert "🗑️ Portföyü Sıfırla" in acikken
    assert acikken["🗑️ Portföyü Sıfırla"].disabled is False

    # Erişim oturumun ortasında kopar.
    geri_al = _break_storage(store)
    try:
        at.run()

        assert not at.exception
        assert "kayıt değiştiren işlemler kapalı" in _texts(at)
        kapaliyken = {b.label: b for b in at.button}
        for yazan in ("➕ Emri Gir / Ekle", "Bakiyeyi Güncelle", "❌ İptal Et",
                      "🗑️ Portföyü Sıfırla"):
            assert yazan in kapaliyken, f"{yazan} çizilmedi"
            assert kapaliyken[yazan].disabled is True, f"{yazan} kapatılmadı"
    finally:
        geri_al()

    store.reset_engine()
    assert store.read_doc(store.PORTFOLIO_KEY) == EMIRLI_PORTFOY


# TP'ye ulaşınca otomatik kapanacak, her iki anahtar kümesini de taşıyan kayıt
# (içeri alınmış eski kayıtlarda görülen biçim).
TP_POZISYONU = {
    "Coin": "Bitcoin (BTC)", "Giriş": 100.0, "Giris": 100.0,
    "Adet": 2.0, "Miktar": 2.0, "Yatırım": 200.0, "Realized": 0.0,
    "Status": "ACTIVE", "TP": 150.0, "SL": 50.0, "Tarih": "2025-12-01",
}


# R4 / QA F10 — Uyarı üreten yol kayıtlara dokunmaz.
#
# Spec 0003 otomatik kapatmayı kaldırdı: fiyat teması artık pozisyonu ve nakdi
# değiştirmiyor, yalnız uyarı gösteriyor. Böylece portföyü yazan tek nokta
# safe_save_portfolio olarak kalıyor ve F10'un koşulu (sarmalayıcıyı atlayan
# bir yazma) ortadan kalkıyor. Bu test o koşulun geri gelmediğini bekler.
def test_stop_alert_does_not_change_records(store, monkeypatch, processed_df):
    store.write_doc(store.ASSETS_KEY, VARLIKLAR)
    baslangic = {"balance": 500.0, "positions": [dict(TP_POZISYONU)]}
    store.write_doc(store.PORTFOLIO_KEY, baslangic)

    at = _make_app(monkeypatch, processed_df)
    # Canlı fiyat stop'un altında: uyarı üretilir. _make_app kendi taklidini
    # kurduğu için bu yama ondan SONRA konur.
    monkeypatch.setattr(data_fetchers, "get_live_price_for_portfolio",
                        lambda *a, **k: 40.0)
    at.run()

    assert not at.exception
    uyarilar = " ".join(str(w.value) for w in at.warning)
    assert "stop teması" in uyarilar
    # Kayıtlar değişmemiştir: pozisyon aktif, bakiye aynı.
    kayitli = store.read_doc(store.PORTFOLIO_KEY)
    assert kayitli == baslangic
    assert at.session_state["portfolio_data"]["positions"][0]["Status"] == "ACTIVE"
    assert at.session_state["portfolio_snapshot"] == kayitli
