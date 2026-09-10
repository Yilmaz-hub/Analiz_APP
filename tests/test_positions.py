"""Pozisyon sınıflandırma testleri — spec 0004 (R4.1).

Kriter eşlemesi:
  AC09  -> test_partially_closed_position_stays_active
  AC10  -> test_fully_closed_position_is_closed
  AC12  -> test_active_position_blocks_deletion
  AC12b -> test_pending_order_blocks_deletion
  AC14  -> test_visibility_and_deletion_share_one_classifier
  AC16c -> test_unreadable_quantity_is_not_treated_as_closed,
           test_broken_record_blocks_deletion,
           test_pending_order_with_unreadable_fields_is_broken (QA F4),
           test_broken_pending_order_blocks_deletion (QA F4)
  AC17  -> test_build_active_rows_survives_zero_investment (yardımcı işlevin
           kayıtları göstermesini engelleyen bölme hatasına düşmediği)
"""
import positions


def _pos(**kwargs):
    base = {"Coin": "Bitcoin (BTC)", "Giriş": 100.0, "Adet": 1.0,
            "Yatırım": 100.0, "Status": "ACTIVE"}
    base.update(kwargs)
    return base


# AC09 — Kalan miktarı sıfırdan büyük olan pozisyon aktiftir.
def test_partially_closed_position_stays_active():
    assert positions.classify_position(_pos(Adet=0.4)) == positions.AKTIF


# AC10 — Kalan miktarı sıfıra inen pozisyon kapalıdır.
def test_fully_closed_position_is_closed():
    assert positions.classify_position(_pos(Adet=0.0)) == positions.KAPALI


def test_pending_order_is_its_own_class():
    assert positions.classify_position(_pos(Status="PENDING")) == positions.BEKLEYEN


def test_auto_closed_position_is_closed():
    # Kapanış kaydında kalan miktar sıfırdır; sınıflandırma miktarı esas alır
    # (G01). Miktarı bitmemiş bir "kapalı" kayıt için bkz.
    # test_partial_sell_marked_confirmed_still_counts_as_active.
    assert positions.classify_position(_pos(Status="CLOSED_TP", Adet=0.0)) == positions.KAPALI


# AC16c / R4.1 — Miktarı okunamayan kayıt kapatılmış sayılmaz.
def test_unreadable_quantity_is_not_treated_as_closed():
    assert positions.classify_position(_pos(Adet="bozuk")) == positions.SORUNLU
    assert positions.classify_position(_pos(Adet=None)) == positions.SORUNLU

    eksik = _pos()
    del eksik["Adet"]
    assert positions.classify_position(eksik) == positions.SORUNLU

    assert positions.classify_position(_pos(Status="???")) == positions.SORUNLU
    assert positions.classify_position(_pos(Coin="")) == positions.SORUNLU
    assert positions.classify_position("bu bir kayıt değil") == positions.SORUNLU


# AC12 — Aktif pozisyon silmeyi engeller.
def test_active_position_blocks_deletion():
    reason = positions.deletion_block_reason("Bitcoin (BTC)", [_pos()])
    assert reason is not None
    assert "aktif pozisyon" in reason


# AC12b — Bekleyen emir silmeyi engeller ve gerekçesi ayrıdır.
def test_pending_order_blocks_deletion():
    reason = positions.deletion_block_reason(
        "Bitcoin (BTC)", [_pos(Status="PENDING")])
    assert reason is not None
    assert "bekleyen emri" in reason


# AC16c — Sorunlu kayıt da silmeyi engeller.
def test_broken_record_blocks_deletion():
    reason = positions.deletion_block_reason("Bitcoin (BTC)", [_pos(Adet="bozuk")])
    assert reason is not None
    assert "okunamayan" in reason


def test_closed_position_does_not_block_deletion():
    assert positions.deletion_block_reason("Bitcoin (BTC)", [_pos(Adet=0.0)]) is None


# AC14 / R4.1 — Görünürlük ve silme engeli aynı sınıflandırıcıdan beslenir.
def test_visibility_and_deletion_share_one_classifier():
    kayitlar = [
        _pos(Coin="Aktif", Adet=2.0),
        _pos(Coin="Kapali", Adet=0.0),
        _pos(Coin="Bekleyen", Status="PENDING"),
        _pos(Coin="Sorunlu", Adet="bozuk"),
    ]
    groups = positions.group_positions(kayitlar)

    assert [p["Coin"] for p in groups[positions.AKTIF]] == ["Aktif"]
    assert [p["Coin"] for p in groups[positions.KAPALI]] == ["Kapali"]
    assert [p["Coin"] for p in groups[positions.BEKLEYEN]] == ["Bekleyen"]
    assert [p["Coin"] for p in groups[positions.SORUNLU]] == ["Sorunlu"]

    # Görünürde aktif olan silinemez, kapalı olan silinebilir: aynı yorum.
    for kind, ad in ((positions.AKTIF, "Aktif"), (positions.BEKLEYEN, "Bekleyen"),
                     (positions.SORUNLU, "Sorunlu")):
        assert positions.blocking_kind_for_asset(ad, kayitlar) == kind
    assert positions.blocking_kind_for_asset("Kapali", kayitlar) is None


def test_reset_block_reason_lists_every_blocking_kind():
    kayitlar = [_pos(Coin="A"), _pos(Coin="B", Status="PENDING"),
                _pos(Coin="C", Adet="bozuk")]
    reason = positions.reset_block_reason(kayitlar)
    assert reason is not None
    for beklenen in ("aktif pozisyon", "bekleyen emir", "okunamayan"):
        assert beklenen in reason

    assert positions.reset_block_reason([_pos(Adet=0.0)]) is None


# AC17 yardımcı — Sıfır yatırımlı kayıt listeyi çökertmez.
def test_build_active_rows_survives_zero_investment():
    rows, total = positions.build_active_rows(
        [_pos(Yatırım=0.0, Adet=1.0)], lambda coin: 120.0)
    assert rows[0]["Kar/Zarar (%)"] == "-"
    assert total == 120.0


def test_build_active_rows_falls_back_to_entry_price():
    rows, total = positions.build_active_rows(
        [_pos(Giriş=50.0, Adet=2.0, Yatırım=100.0)], lambda coin: 0)
    assert rows[0]["Değer ($)"] == 100.0
    assert total == 100.0


# AC16c / QA F4 — Alanları okunamayan bekleyen emir sorunlu sayılır.
def test_pending_order_with_unreadable_fields_is_broken():
    # QA reprosu: yalnız Coin ve Status içeren bekleyen kayıt.
    assert positions.classify_position(
        {"Coin": "Solana (SOL)", "Status": "PENDING"}) == positions.SORUNLU

    for eksik in ("Giriş", "Adet", "Yatırım"):
        kayit = _pos(Status="PENDING")
        del kayit[eksik]
        assert positions.classify_position(kayit) == positions.SORUNLU, eksik

    kayit = _pos(Status="PENDING", Yatırım="okunamaz")
    assert positions.classify_position(kayit) == positions.SORUNLU


# QA F4 — Aktif kayıtta da tabloda kullanılan alanlar doğrulanır.
def test_active_position_with_unreadable_amount_is_broken():
    assert positions.classify_position(_pos(Yatırım="okunamaz")) == positions.SORUNLU
    assert positions.classify_position(_pos(Giriş=None)) == positions.SORUNLU


# QA F4 — Kapanmış kayıt listelenmediği için alan doğrulaması aranmaz.
def test_closed_position_without_amount_stays_closed():
    kapali = _pos(Adet=0.0)
    del kapali["Yatırım"]
    assert positions.classify_position(kapali) == positions.KAPALI


# AC16c / QA F4 — Bozuk bekleyen emir varlığın silinmesini engeller.
def test_broken_pending_order_blocks_deletion():
    bozuk = {"Coin": "Bitcoin (BTC)", "Status": "PENDING"}
    reason = positions.deletion_block_reason("Bitcoin (BTC)", [bozuk])
    assert reason is not None
    assert "okunamayan" in reason


# AC09 / spec 0003 uyumu — Kısmi satışta CLOSED_CONFIRMED yazılsa bile kalan
# miktarı olan pozisyon aktiftir ve varlığın silinmesini engeller.
def test_partial_sell_marked_confirmed_still_counts_as_active():
    kismi = _pos(Status="CLOSED_CONFIRMED", Adet=0.4)
    assert positions.classify_position(kismi) == positions.AKTIF
    assert positions.deletion_block_reason("Bitcoin (BTC)", [kismi]) is not None


# AC10 / spec 0003 uyumu — Miktarı biten CLOSED_CONFIRMED kaydı kapalıdır.
def test_fully_sold_confirmed_position_is_closed():
    tam = _pos(Status="CLOSED_CONFIRMED", Adet=0.0)
    assert positions.classify_position(tam) == positions.KAPALI
    assert positions.deletion_block_reason("Bitcoin (BTC)", [tam]) is None
