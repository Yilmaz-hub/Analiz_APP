# Spec: 0022 — Varlığa Göre Sinyal Kuralı (Ekran)

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-11).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi onayı (2026-10-11, "Evet varliga gore kural secen yeni spec yapilacak"; BTC/ETH/altın C, altcoin Donchian).

## Bağlam
0019–0021 ile kurallar varlık sınıfına göre belirlendi (C, kanal 50/20, al-tut) ve portföy testi bunlarla yapılıyor; ekrandaki günlük sinyal ise hâlâ V1'di. Kanıt: 0019 (C), 0020 (kanal), 0021 (hisse al-tut). Sanal takip (V1 + yöntem adayı) değişmeden sürer.

## Requirements
- **R01 — Kural:** Günlük sinyal son kapanmış mumdan, `asset_method`'a göre üretilir: `TREND_SMA` (BTC/ETH, türevleri, sabit coinler, altın): V1 SAT → SAT; V1 AL ve kapanış > 200 günlük basit ortalama → AL; aksi BEKLE (200 gün yoksa AL yok). `KANAL` (altcoin): kapanış önceki 50 günün en yükseğinin üstünde AL, önceki 20 günün en düşüğünün altında SAT, aksi BEKLE; V1 kullanılmaz. `TUT` (hisse): zamanlama sinyali yok (BEKLE), not yazılır. Kuralı olmayan varlık (döviz) V1'i aynen kullanır. Stop ve pozisyon yönetimi aynıdır (2,5 ATR, zararlı çıkış sonrası bekleme).
- **R02 — Ekran:** Günlük görünümde karar paneli R01 sinyalini kullanır ve "Sinyal kuralı: … → … (V1 kararı: …)" satırını yazar; yan paneldeki günlük kartın altında aynı satır görünür. Veri geçersizse (kapanmış mum doğrulanamadı) kural hesaplanmaz, mevcut davranış sürer.
- **R03 — Seçim:** Karar panelinde "Sinyal kuralı" seçimi: "Varlığa göre" (varsayılan) ya da "V1 (eski)". Seçim kaydedilir ve tüm varlıklar için geçerlidir. "V1 (eski)" seçiliyken önceki davranış (aktif strateji / V1) aynen sürer.

## Acceptance Criteria
- [ ] **AC01 — C, R01:** BTC/ETH/altında V1 AL + ortalama üstü AL, ortalama altı BEKLE, V1 SAT SAT; 200 gün yoksa AL yok.
- [ ] **AC02 — Kanal, R01:** Altcoinde kırılım AL, kanal altı SAT, V1'den bağımsız.
- [ ] **AC03 — Hisse/döviz, R01:** Hissede BEKLE + al-tut notu; dövizde V1 aynen.
- [ ] **AC04 — Seçim, R03:** Varsayılan varlığa göre; V1'e çevrilince kaydedilir; geçersiz değer reddedilir.
- [ ] **AC05 — Ekran, R02/R03:** Günlük görünümde karar paneli ve yan panel kural satırını yazar; "V1 (eski)" seçilince satır kalkar.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü; yayında günlük karar paneli varlığın kuralını gösterir

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | — |
