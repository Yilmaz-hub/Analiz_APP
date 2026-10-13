# Spec: 0025 — Portföy Testinde Başlangıç ve Isınma, Backtest'te Kalan Nakitle Alım

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-13).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi kararı (2026-10-13): "2, 3 onay. 1. Tüm varlıkların işlem
> yapılabildiği ilk günü beklemeye gerek yok … BTC ETH gibi varlıklar ile normalde ne zaman teste başlıyorsan o zaman
> başla … 2024 öncesinde değeri yoksa işlem yapılamadı diyerek nakit sayıldı diye uyarı ver".

## Intent
Portföy testi (ağırlıklarla, varsayılan sepet) ekranda yıllık %6–8 gösterirken aynı veride doğru kurulmuş kıyas
%17 veriyordu. Neden: (1) test, en uzun geçmişli varlığın (altın, Yahoo ~4 yıl) hazır olduğu günden başlıyordu;
kripto 1000 günlük veride ancak bir yıl sonra işlem yapabildiği için paranın %80'i bir yıl boşta kalıyor, al-tut ise
kriptoyu ilk mumda (ısınma içinde) alıyordu; (2) altcoinler V1'in ilk kararından önce (ısınmada) işlem yapıyordu.
Ayrıca tek varlık backtest'inde işlem tutarı ≈ sermaye iken ilk zararlı işlemden sonra tüm alımlar "nakit yetersiz"
ile duruyordu. Başarı: kullanıcı portföyü ve tek varlığı dürüst bir pencerede, anlaşılır uyarılarla görür.

## Requirements
- **R01 — Başlangıç:** Portföy testi referans varlıkların en erken işlem gününden başlar: seçilmişse BTC/ETH;
  yoksa seçilen kriptolar; yoksa tüm varlıklar. Daha erken hazır olan varlık (ör. altın) beklenmez, öncesi teste
  girmez. Daha geç hazır olan varlığın payı o güne kadar nakittir; ekranda "X: <tarih> öncesinde verisi ya da
  göstergesi olmadığı için işlem yapılamadı; o tarihe kadar payı (%w) nakit sayıldı." uyarısı çıkar. Al-tut
  karşılaştırması ve stopsuz tut alımı da o varlığı ilk işlem gününde alır. Tut yöntemli varlık ısınma beklemez
  (ilk mumda hazır).
- **R02 — Isınma:** Portföyde zamanlamalı varlıkların kararları V1'in ilk karar gününden (bar 199) önce kullanılmaz
  (0023'ün tek varlık kuralıyla aynı).
- **R03 — Kalan nakitle alım:** Ortak backtest motorunda (`run_v1_strategy_backtest`: tek varlık backtest'i, sanal
  takip işlemleri, aday ve rapor karşılaştırmaları) AL günü nakit, işlem tutarına (komisyon dahil) yetmiyor ama en az
  yarısını karşılıyorsa alım eldeki nakitle yapılır; yarısından azsa alım "nakit yetersiz" olarak engellenir
  (portföy testindeki kuralla aynı).

## Constraints
- Para/yüzde `Decimal`. `technical_analysis.py` CRLF satır sonuyla saklanır.
- **Kapsam dışı:** kripto için 1000 günden uzun geçmiş çekmek (ayrı spec önerisi); ağırlık sınırı (0024).

## Context
- Modüller: `portfolio_backtest.py` (`portfolio_start`, `run_portfolio(available_from=…)`, `_hold`),
  `portfolio_ui.py` (`_run`), `technical_analysis.py` (`_affordable`, `run_v1_strategy_backtest`).
- Kanıt (2026-10-13, canlı veri, kullanıcının ağırlıkları BTC/ETH 20, SOL/XRP 10, AVAX 2, altın 20, HBAR/LINK/ADA 5,
  RENDER 3): önce 2023-08-02 → strateji yıllık %7,7 / al-tut %13,5; sonra 2024-08-05 → strateji %17,1 (düşüş −%20)
  / al-tut %18,5 (−%56); RENDER uyarısı 2025-02-11. Tek varlık BTC, işlem tutarı 9.900 / sermaye 10.000: önce 1
  işlem −%10,6 (sonraki alımlar engelli).

## Acceptance Criteria
- [ ] **AC01 — Başlangıç kuralı (R01):** BTC/ETH varken başlangıç onların en erken işlem günü; erken hazır altın beklenmez; geç hazır olanlar listelenir; referans yoksa en erken gün.
- [ ] **AC02 — Geç varlık nakit (R01):** Geç hazır varlığın payı o güne kadar nakittir; al-tut ve tut alımı o gün yapılır.
- [ ] **AC03 — Panel (R01/R02):** Portföy testi BTC'nin ilk işlem gününden başlar, geç altcoin için tarih ve ağırlıklı uyarı çıkar, hiçbir işlem varlığın ilk işlem gününden önce açılmaz.
- [ ] **AC04 — Kalan nakit (R03):** İşlem tutarı ≈ sermaye iken zararlı çıkıştan sonra kalan nakitle alım sürer; nakit yarıdan azsa engellenir.

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
| Kaçan hata | 0019 QA düşük notu ("al-tut global en erken günden alıyor") kullanıcıya %6 vs %17 farkı olarak ulaştı |
