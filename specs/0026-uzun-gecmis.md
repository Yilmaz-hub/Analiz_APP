# Spec: 0026 — Backtest ve Portföy Testinde Uzun Geçmiş (2018'den) ve Karar Önbelleği

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-13).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi onayı (2026-10-13, 4. madde "Onaylıyorum").
> Bağımlılık: 0025 ([PR #39](https://github.com/yilmaz-hub/analiz_app/pull/39)); bu dal onun üzerine açıldı.

## Intent
Uygulamanın günlük verisi kriptoda son 1000 gün; 200 günlük ısınmadan sonra backtest ve portföy testi ~2,2 yıllık
tek bir döneme (2024-08 → bugün) bağlı kalıyor. Kullanıcı stratejiyi farklı piyasa dönemlerinde (2018 düşüşü,
2020–21 yükselişi, 2022 düşüşü, 2023–26) görmek istiyor; önceki deneylerin %50'ye varan yıllık sonuçlarının bu
uzun dönemden geldiği ekranda doğrulanabilmeli. Başarı: backtest ve portföy testinde "2018'den" dönemi seçilebilir,
ilk çalıştırma kabul edilebilir sürede biter, sonrakiler hızlıdır; grafik ve sinyal etkilenmez.

## Requirements
- **R01 — Uzun veri:** "2018'den" döneminde günlük veri: kripto Binance 2018-01-01'den (1000'lik sayfalar;
  `data-api.binance.vision`, sonra `api.binance.com`); ons altın GC=F, hisse ve USD/TRY Yahoo 2017-01-01'den; gram
  altın = GC=F × USD/TRY ÷ 31,1035. Uzun veri alınamazsa uygulamanın günlük verisi kullanılır ve "uzun geçmiş
  alınamadı" notu yazılır. Uzun veri saatlik önbelleklenir.
- **R02 — Dönem seçimi:** Günlük backtest'te ve portföy testinde "Test dönemi": "2018'den (uzun geçmiş)"
  (varsayılan) / "Son ~2 yıl (uygulamanın günlük verisi)". Backtest kullanılan veri aralığını yazar.
  Grafik, sinyal, sanal takip ve diğer paneller 1000 günlük veride kalır.
- **R03 — Karar önbelleği:** Uzun dönemde V1 kararları depoda saklanır (anahtar: sembol + veri türü [dolara
  çevrilmiş] + parmak izi [ağırlıklar, ML, kararlılık penceresi]); sonraki çalıştırmalarda yalnız yeni günler
  hesaplanır (`build_v1_decisions(only_last=N)`, tam hesapla birebir aynı). Son mum saklanmaz (sağlayıcıda
  kapanmamış olabilir). Verinin ilk günü değişirse tam hesaplanır. Depo okunamaz/yazılamazsa önbelleksiz hesaplanır.
  İlk hesapta "N günün kararı hesaplanıyor (ilk seferde ~1–2 dakika)" notu çıkar.
- **R04 — 0025 kuralları uzun dönemde de geçerli:** başlangıç BTC/ETH'nin ilk işlem günü; geç başlayan varlığın payı
  nakit + uyarı; ısınmada işlem yok.

## Constraints
- Para/yüzde `Decimal`; `app.py` CRLF. Testler ağa çıkmaz (`tests/conftest.py::offline_history`).
- **Kapsam dışı:** grafik/sinyal verisini uzatmak; haftalık ve 4 saatlik periyotlar; Binance dışı kripto kaynakları
  (OKX) için uzun veri.

## Context
- Modüller: `history.py` (yeni: `fetch_binance_daily`, `fetch_yahoo_daily`, `load_history`, `cached_v1_decisions`),
  `app.py` (📊 Backtest bloğu), `portfolio_ui.py` (`_run`, form).
- Ölçüm (2026-10-13): `build_v1_decisions` 1000 gün 15 sn, 3200 gün 84 sn (varlık başına) → önbellek gerekli.
  Veri: BTC 3208 gün (2018-01-01'den), GC=F/hisse/kur ~2460 gün (2017-01'den).

## Acceptance Criteria
- [ ] **AC01 — Binance sayfalama (R01):** 2018-01-01'den 1000'lik sayfalar birleşir; ilk adres hata verirse sıradaki denenir; kripto olmayanda None.
- [ ] **AC02 — Dönem ve geri dönüş (R01/R02):** "2018'den" uzun veriyi, "son ~2 yıl" uygulama verisini kullanır; uzun veri yoksa uygulama verisi + not.
- [ ] **AC03 — Gram altın (R01):** uzun geçmişte gram altın = ons × kur ÷ 31,1035; ons GC=F; hisse kendi kodu.
- [ ] **AC04 — Önbellek (R03):** ikinci çalıştırmada yalnız yeni günler hesaplanır, sonuç tam hesapla aynı; farklı ağırlık ayrı kayıt; ilk gün değişince tam hesap; depo hatasında önbelleksiz.
- [ ] **AC05 — Ekran (R02):** günlük backtest'te dönem seçimi (varsayılan 2018'den) kullanılan veriyi ve aralığını belirler.

## Definition of Done
- [ ] Her kriter bir pytest testi; yeşil pipeline (`pytest`, `-m browser`); bağımsız QA kabulü; squash-merge.

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | — |
| Regresyon sayısı | 0 |
| Kaçan hata | — |
