"""Kayıt gösterme performansı — spec 0004, AC17 (P1 onaylı).

Ölçüm, kayıtların okunmasından listenin hazır olmasına kadar geçen süredir:
`load_assets` + `load_portfolio` + sınıflandırma + aktif pozisyon tablosunun
kurulması. Canlı piyasa fiyatı çekme süresi ölçüme dahil edilmez (fiyat
sağlayıcı taklit edilir).

Örnek: 100 varlık ve toplam 100 aktif pozisyon. Aynı koşullarda 10 ölçüm
yapılır ve **her biri** 5 saniye sınırını sağlamalıdır.

Kriter eşlemesi:
  AC17 -> test_hundred_assets_and_positions_render_under_five_seconds
"""
import time

import assets as assets_module
import portfolio as portfolio_module
import positions

VARLIK_SAYISI = 100
POZISYON_SAYISI = 100
OLCUM_SAYISI = 10
SINIR_SANIYE = 5.0


def _ornek_kayitlar():
    varliklar = {f"Varlık {i:03d}": f"SYM{i:03d}-USD" for i in range(VARLIK_SAYISI)}
    adlar = list(varliklar)
    pozisyonlar = [
        {
            "Coin": adlar[i % VARLIK_SAYISI], "Giriş": 100.0 + i, "Adet": 1.5,
            "Yatırım": 150.0 + i, "Realized": 0.0, "Status": "ACTIVE",
            "Tarih": "2026-01-01",
        }
        for i in range(POZISYON_SAYISI)
    ]
    return varliklar, {"balance": 10000.0, "positions": pozisyonlar}


# AC17 — 100 varlık ve toplam 100 aktif pozisyon, 10 ölçümün her birinde ≤ 5 sn.
def test_hundred_assets_and_positions_render_under_five_seconds(store):
    varliklar, portfoy = _ornek_kayitlar()
    assets_module.save_assets(varliklar)
    portfolio_module.save_portfolio(portfoy)

    sabit_fiyat = lambda coin: 125.0  # canlı fiyat çekimi ölçüme dahil değil

    sureler = []
    for _ in range(OLCUM_SAYISI):
        baslangic = time.perf_counter()

        yuklenen_varliklar = assets_module.load_assets()
        yuklenen_portfoy = portfolio_module.load_portfolio()
        gruplar = positions.group_positions(yuklenen_portfoy["positions"])
        satirlar, _toplam = positions.build_active_rows(
            gruplar[positions.AKTIF], sabit_fiyat)

        sureler.append(time.perf_counter() - baslangic)

        assert len(yuklenen_varliklar) == VARLIK_SAYISI
        assert len(satirlar) == POZISYON_SAYISI

    assert len(sureler) == OLCUM_SAYISI
    en_uzun = max(sureler)
    assert all(sure <= SINIR_SANIYE for sure in sureler), (
        f"10 ölçümün en uzunu {en_uzun:.3f} sn, sınır {SINIR_SANIYE} sn"
    )
    print(f"AC17 ölçümü: en uzun {en_uzun:.3f} sn, ortalama "
          f"{sum(sureler) / len(sureler):.3f} sn ({OLCUM_SAYISI} ölçüm)")
