# Spec: 0020 — Altcoinlerde Kanal Kırılımı (Donchian 50/20)

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-11).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi onayı (2026-10-11, "Onaylandi altcoinler icin dochian"; varsayılan ağırlıklar "Varsayilan olsun").

## Kanıt (deney; ML'siz V1; Binance günlük; komisyon %0,1; yöntemler ve kabul ölçütü sonuç görülmeden yazıldı)
- **Ön kayıt:** Altcoinler (BTC/ETH dışı 19 kripto) için M0 = C (spec 0019), M1 = kanal kırılımı 50/20, M2 = altcoin/BTC göreli güç, M3 = C + göreli güç. Dönemler: geliştirme (ilk işlem günü → 2022-06-30) ve doğrulama (2022-07-01 → bugün). Ölçüt: M0'ı iki dönemde de medyan yıllık getiride geçmek ve medyan en büyük düşüşü 5 puandan fazla kötüleştirmemek.
- **Sonuç (tek varlık, tüm sermaye, medyan):**

| Yöntem | Geliştirme: yıllık / düşüş | Doğrulama: yıllık / düşüş |
|---|---|---|
| M0 C | %34,6 / −%70 | %3,3 / −%72 |
| **M1 kanal 50/20** | **%63,7 / −%61** | **%5,7 / −%65** |
| M2 göreli güç | −%9,0 / −%77 | −%2,1 / −%64 |
| M3 C + göreli güç | %6,9 / −%68 | %5,2 / −%62 |

- **Portföy (P1 ağırlıkları + altın %20; BTC/ETH/altın C, altcoinler kanal):** 2018-11→ yıllık %61,9 → %62,3, düşüş −%66 → −%57; 2021-05→ %19,6 → %24,1, −%44 → −%33; 2022-07→ %25,8 → %21,0, −%39 → −%27; 2024-08→ %20,2 → %19,2, −%27 → −%25. Tüm varlıklarda kanal daha kötü (BTC/ETH'de C kalır).
- **Büyük / küçük altcoin ayrımı:** iki grupta da kanal C'yi ancak coinlerin yaklaşık yarısında geçiyor; ortak etki düşüşün azalması. 9–10 coinlik gruplarla ayrı kural güvenilir değil; tüm altcoinler tek grup.
- **Altın:** RSI(2) dip, kanal ve C al-tut'u geçemedi; altın C ile kalır.
- **Sınırlar:** geçmiş veri; bugün var olan coinler (batanlar yok); kayma yok; gelecek getirisi değildir.

## Requirements
- **R01 — Kanal kuralı:** Yeni aday kural `KANAL` (giriş) / `KANAL` (çıkış): kapanış, önceki `entry` günün (varsayılan 50) en yükseğinin üstündeyse AL; kapanış, önceki `exit` günün (varsayılan 20) en düşüğünün altındaysa SAT; değilse BEKLE. Her gün yalnız o güne kadarki veri kullanılır; her pencere kendi uzunluğu dolmadan o kararı vermez (SAT, giriş penceresinden önce oluşabilir; pozisyon yokken etkisizdir). V1 kararı kullanılmaz. İşletme kuralları aynıdır (ertesi açılış, 2,5 ATR stop, zararlı çıkışta 2 gün bekleme). Onaylı giriş ve çıkış kuralları listesine eklenir.
- **R02 — Varlık sınıfına göre yöntem:** Altcoin = kripto piyasası ve tabanı BTC ya da ETH olmayan varlık. Altcoinlerin yöntemi `KANAL`, diğer tüm varlıkların C (`TREND_SMA`). Tek yer: `strategy_candidates.method_candidate`.
- **R03 — Sanal takip:** İleri takip koşucusu, aktif stratejinin (V1) yanında her varlık için R02'deki yöntem adayını izler (altcoinlerde kanal, diğerlerinde C). Aday ön kaydı piyasa ve kural başına bir kez yapılır. Aktif strateji değişmez. Altcoinlerde önceki C kayıtları silinmez, yalnız yeni gün eklenmez.
- **R04 — Portföy testi:** Panelde "Altcoinlerde kanal kırılımı (50/20)" seçeneği (varsayılan açık). Açıkken her varlık R02 yöntemiyle, kapalıyken tümü C ile test edilir. Varlık tablosunda her varlığın yöntemi görünür.

## Acceptance Criteria
- [ ] **AC01 — Kanal kararı, R01:** Kapanış önceki 50 günün en yükseğini aşınca AL, önceki 20 günün en düşüğünün altına inince SAT, arada BEKLE; pencere dolmadan BEKLE; V1 SAT/AL kararı etkilemez; ayarlar kararı değiştirir; kural onaylıdır.
- [ ] **AC02 — Yöntem seçimi, R02:** SOL/XRP/DOGE/PEPE için kanal; BTC, ETH, ons altın ve hisse için C.
- [ ] **AC03 — Sanal takip, R03:** Koşucu bir altcoin için V1 ve kanal sürümlerini, BTC için V1 ve C sürümlerini kaydeder.
- [ ] **AC04 — Portföy, R04:** Seçenek açıkken altcoin işlemleri kanal kararlarıyla, kapalıyken C kararlarıyla üretilir; ekranda yöntem sütunu görünür.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü; yayında portföy testi seçenekle çalışır

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | — |
