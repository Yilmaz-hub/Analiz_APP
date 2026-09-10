"""Varlık listesi iş kuralları — spec 0004.

Kriter eşlemesi:
  AC02  -> test_asset_without_position_is_found_after_reload
  AC07  -> test_missing_document_does_not_create_records
  AC08  -> test_single_asset_count_stays_one_after_reload
  AC09  -> test_partially_closed_position_blocks_delete
  AC10  -> test_fully_closed_position_keeps_asset_in_list
  AC11  -> test_clean_asset_is_deleted_and_stays_deleted
  AC11b -> test_last_asset_cannot_be_deleted
  AC11c -> test_reset_blocked_by_active_position, test_reset_requires_confirmation
  AC11d -> test_portfolio_reset_blocked_by_records, test_portfolio_reset_requires_confirmation
  AC12  -> test_asset_with_active_position_cannot_be_deleted
  AC12b -> test_asset_with_pending_order_cannot_be_deleted
  AC12c -> test_no_orphan_active_records_after_any_delete_path
  AC15  -> test_invalid_add_leaves_records_untouched
  AC15b -> test_duplicate_name_does_not_overwrite_existing_asset
  AC16b -> test_write_failure_leaves_stored_records_untouched
"""
import pytest

import assets as assets_module
import portfolio as portfolio_module
import positions

IKI_VARLIK = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"}


def _pos(coin="Bitcoin (BTC)", **kwargs):
    base = {"Coin": coin, "Giriş": 100.0, "Adet": 1.0, "Yatırım": 100.0,
            "Status": "ACTIVE"}
    base.update(kwargs)
    return base


# AC02 — Pozisyonu olmayan varlık yeniden açılışta bulunur.
def test_asset_without_position_is_found_after_reload(store):
    ok, _, updated = assets_module.add_asset({}, "Pound", "GBPUSD=X")
    assert ok
    assets_module.save_assets(updated)

    store.reset_engine()

    assert assets_module.load_assets() == {"Pound": "GBPUSD=X"}


# AC07 — Kayıt yokken kendiliğinden kayıt oluşmaz.
def test_missing_document_does_not_create_records(store):
    assert assets_module.load_assets() == {}
    assert store.read_doc(store.ASSETS_KEY) is None
    assert portfolio_module.load_portfolio()["positions"] == []
    assert store.read_doc(store.PORTFOLIO_KEY) is None


# AC08 — Tek varlıklı liste yeniden açılışta bir varlık olarak kalır.
def test_single_asset_count_stays_one_after_reload(store):
    assets_module.save_assets({"Pepe": "PEPE-USD"})
    store.reset_engine()
    assert len(assets_module.load_assets()) == 1


# AC15 — Geçersiz ekleme mevcut kayıtları değiştirmez.
@pytest.mark.parametrize("ad,sembol", [("", "BTC-USD"), ("Yeni", ""), (None, None)])
def test_invalid_add_leaves_records_untouched(ad, sembol):
    ok, msg, updated = assets_module.add_asset(IKI_VARLIK, ad, sembol)
    assert ok is False
    assert updated == IKI_VARLIK
    assert "boş olamaz" in msg


# AC15b — Kayıtlı adla ekleme mevcut kaydı sessizce ezmez.
def test_duplicate_name_does_not_overwrite_existing_asset():
    ok, msg, updated = assets_module.add_asset(IKI_VARLIK, "Bitcoin (BTC)", "BASKA-KOD")
    assert ok is False
    assert "zaten kayıtlı" in msg
    assert updated == IKI_VARLIK
    assert updated["Bitcoin (BTC)"] == "BTC-USD"


def test_duplicate_name_check_ignores_letter_case():
    ok, _, updated = assets_module.add_asset(IKI_VARLIK, "bitcoin (btc)", "BASKA-KOD")
    assert ok is False
    assert updated == IKI_VARLIK


# AC11 — Açıkça silinen varlık yeniden açılışta listede bulunmaz.
def test_clean_asset_is_deleted_and_stays_deleted(store):
    assets_module.save_assets(IKI_VARLIK)
    ok, _, updated = assets_module.delete_asset(assets_module.load_assets(),
                                                "Ethereum (ETH)", [])
    assert ok
    assets_module.save_assets(updated)

    store.reset_engine()

    kalan = assets_module.load_assets()
    assert "Ethereum (ETH)" not in kalan
    assert "Bitcoin (BTC)" in kalan


# AC11b — Listedeki son varlık silinemez.
def test_last_asset_cannot_be_deleted():
    tek = {"Bitcoin (BTC)": "BTC-USD"}
    ok, msg, updated = assets_module.delete_asset(tek, "Bitcoin (BTC)", [])
    assert ok is False
    assert "en az 1 varlık" in msg
    assert updated == tek


# AC12 — Aktif pozisyonlu varlık silinemez; kayıtlar korunur.
def test_asset_with_active_position_cannot_be_deleted():
    kayitlar = [_pos()]
    ok, msg, updated = assets_module.delete_asset(IKI_VARLIK, "Bitcoin (BTC)", kayitlar)
    assert ok is False
    assert "aktif pozisyon" in msg
    assert updated == IKI_VARLIK
    assert kayitlar == [_pos()]


# AC09 — Kısmen kapatılmış pozisyonun varlığı silinemez.
def test_partially_closed_position_blocks_delete():
    ok, msg, updated = assets_module.delete_asset(
        IKI_VARLIK, "Bitcoin (BTC)", [_pos(Adet=0.3)])
    assert ok is False
    assert "aktif pozisyon" in msg
    assert updated == IKI_VARLIK


# AC10 — Tamamen kapanmış pozisyonun varlığı listede kalır (silme yapılmadan).
def test_fully_closed_position_keeps_asset_in_list(store):
    assets_module.save_assets(IKI_VARLIK)
    portfolio_module.save_portfolio({"balance": 100.0, "positions": [_pos(Adet=0.0)]})

    store.reset_engine()

    assert "Bitcoin (BTC)" in assets_module.load_assets()


# AC12b — Bekleyen emirli varlık silinemez, kilitli tutar korunur.
def test_asset_with_pending_order_cannot_be_deleted():
    emir = _pos(Status="PENDING", Yatırım=750.0)
    ok, msg, updated = assets_module.delete_asset(IKI_VARLIK, "Bitcoin (BTC)", [emir])
    assert ok is False
    assert "bekleyen emri" in msg
    assert updated == IKI_VARLIK
    assert emir["Yatırım"] == 750.0


# AC16c — Sorunlu kayıtlı varlık silinemez ve kayıt korunur.
def test_asset_with_broken_record_cannot_be_deleted():
    bozuk = _pos(Adet="okunamaz")
    ok, msg, updated = assets_module.delete_asset(IKI_VARLIK, "Bitcoin (BTC)", [bozuk])
    assert ok is False
    assert "okunamayan" in msg
    assert updated == IKI_VARLIK
    assert bozuk["Adet"] == "okunamaz"


# AC11c — Aktif pozisyon varken liste sıfırlaması çalışmaz.
def test_reset_blocked_by_active_position():
    ok, msg, updated = assets_module.reset_assets(IKI_VARLIK, [_pos()], confirmed=True)
    assert ok is False
    assert "aktif pozisyon" in msg
    assert updated == IKI_VARLIK


# AC11c — Sıfırlama her durumda ayrıca onay ister.
def test_reset_requires_confirmation():
    ok, msg, updated = assets_module.reset_assets(IKI_VARLIK, [], confirmed=False)
    assert ok is False
    assert "onay" in msg
    assert updated == IKI_VARLIK

    ok, _, updated = assets_module.reset_assets(IKI_VARLIK, [], confirmed=True)
    assert ok is True
    assert updated == assets_module.default_assets()


# AC11d — Portföy sıfırlaması engelli kayıt varken çalışmaz.
@pytest.mark.parametrize("kayit", [_pos(), _pos(Status="PENDING"), _pos(Adet="bozuk")])
def test_portfolio_reset_blocked_by_records(kayit):
    portfoy = {"balance": 500.0, "positions": [kayit]}
    ok, msg, sonuc = portfolio_module.reset_portfolio(portfoy, confirmed=True)
    assert ok is False
    assert "Sıfırlama yapılamaz" in msg
    assert sonuc == portfoy
    assert sonuc["positions"] == [kayit]


# AC11d — Portföy sıfırlaması ayrıca onay ister.
def test_portfolio_reset_requires_confirmation():
    portfoy = {"balance": 500.0, "positions": [_pos(Adet=0.0)]}
    ok, msg, sonuc = portfolio_module.reset_portfolio(portfoy, confirmed=False)
    assert ok is False
    assert "onay" in msg
    assert sonuc == portfoy

    ok, _, sonuc = portfolio_module.reset_portfolio(portfoy, confirmed=True)
    assert ok is True
    assert sonuc["positions"] == []


# AC12c — Hiçbir silme yolu, karşılığı olmayan aktif/bekleyen kayıt bırakmaz.
def test_no_orphan_active_records_after_any_delete_path():
    kayitlar = [
        _pos(coin="Bitcoin (BTC)"),
        _pos(coin="Ethereum (ETH)", Status="PENDING"),
        _pos(coin="Pepe", Adet=0.0),
    ]
    liste = dict(IKI_VARLIK)
    liste["Pepe"] = "PEPE-USD"

    sonuclar = []
    for ad in list(liste.keys()):
        sonuclar.append(assets_module.delete_asset(liste, ad, kayitlar))
    ok, _, sifirlanan = assets_module.reset_assets(liste, kayitlar, confirmed=True)
    sonuclar.append((ok, "", sifirlanan))

    for _, _, kalan_liste in sonuclar:
        gruplar = positions.group_positions(kayitlar)
        for kayit in gruplar[positions.AKTIF] + gruplar[positions.BEKLEYEN]:
            assert positions.asset_name(kayit) in kalan_liste


# AC16b — Yazma başarısız olduğunda depodaki kayıtlar değişmez.
def test_write_failure_leaves_stored_records_untouched(store, monkeypatch):
    assets_module.save_assets(IKI_VARLIK)

    gercek_yazma = store.write_doc

    def kirik_yazma(key, payload):
        raise store.StorageAccessError("test")

    store.write_doc = kirik_yazma
    try:
        with pytest.raises(store.StorageAccessError):
            assets_module.save_assets({"Sadece": "BIR"})
    finally:
        store.write_doc = gercek_yazma

    assert assets_module.load_assets() == IKI_VARLIK
