# Spec: 0023 — Tek Varlık Backtest'i Varlığın Kuralıyla

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-11).
> Durum: **ONAYLANDI — GELİŞTİRİLECEK (devir).** Kaynak: Takım Yöneticisi, 2026-10-11 ("yapılacak spec'i GitHub'a yükle,
> başka bir developer geliştirmeyi tamamlasın"). Spec 0022 sonrası açık kalan eksik.
> **Bağımlılık:** spec 0022 ([PR #37](https://github.com/yilmaz-hub/analiz_app/pull/37)) `main`'e alınmış olmalı
> (`signal_rules.py`, `signal_mode`). Bu dal `main`'den açıldı; #37 birleşince `main` bu dala alınır (merge).

## Intent
Kullanıcı BTC'ye bakarken ekrandaki sinyal artık varlığın kuralından (0022) geliyor: BTC/ETH/altın C, altcoinler
kanal kırılımı 50/20, hisseler al-tut. Ama ekrandaki "📊 Backtest: Strateji Performansı" hâlâ V1 ile çalışıyor;
kullanıcı BTC'de backtest yaptığında gördüğü sonuç, ekrandaki sinyalin kuralının sonucu değil. Başarı: günlük
backtest, karar panelindeki sinyalle **aynı kuralı** test eder; hangi kuralın test edildiği ekranda yazar.

## Requirements
- **R01 — Kural kararları:** Günlük (`1d`) backtest, `signal_rules.signal_mode()` "VARLIK" iken varlığın kuralının
  kararlarıyla çalışır (`strategy_candidates.asset_method(symbol)`):
  - `TREND_SMA` → `sc.candidate_decisions(df_view, v1_decisions, sc.trend_candidate(market), None)`
  - `KANAL` → `sc.candidate_decisions(df_view, {}, sc.channel_candidate(market), None)`; **yalnız ilk V1 karar
    gününden (bar 199) itibaren** kararlar tutulur (deney ve portföy testiyle aynı ısınma; al-tut karşılaştırması
    ve teşhis aynı başlangıcı kullanır).
  - `TUT` (hisse) → zamanlama backtest'i yapılmaz; "Bu varlıkta yöntem al-tut; zamanlama testi yok" bilgisi ve
    yalnız al-tut sonucu (mevcut `performance_ui.hold_comparison` ile aynı dönem) gösterilir.
  - `None` (ör. döviz) ya da mod "V1" → bugünkü V1 davranışı aynen.
- **R02 — Etiket:** Backtest bilgi kutusu test edilen kuralı yazar (ör. "Günlük test — kural: Kanal kırılımı
  50/20 …"). Bugünkü sabit "Günlük V1 testi…" metni yalnız V1 kullanıldığında görünür.
- **R03 — Bağlı paneller:** Al-tut karşılaştırması (`render_hold_comparison`), teşhis (`render_diagnostics`),
  Güvenilir Performans Raporu (`performance_ui.render_report_panel` → `comparison_from_backtest`) ve kâr koruma
  (`protection_ui`) **kural kararlarını** kullanır (`perf_report_source["decisions"]`).
- **R04 — Aday paneli V1'de kalır:** Aday değerlendirmesi (`candidate_ui._render_candidate_panel` →
  `evaluate_candidates`) referans olarak **V1 kararlarına** ihtiyaç duyar. `perf_report_source`'a ayrıca
  `"v1_decisions"` eklenir; aday paneli `source.get("v1_decisions", source["decisions"])` kullanır.
- **R05 — İşletme kuralları değişmez:** `run_v1_strategy_backtest` motoru (ertesi açılış, 2,5 ATR sabit stop,
  zararlı çıkış sonrası 2 gün bekleme, maliyetler) tüm kurallarda aynen kullanılır; yalnız karar sözlüğü değişir.

## Constraints
- Para/yüzde `Decimal` (docs/conventions.md). `app.py` **CRLF** satır sonuyla saklanır; düzenlerken korunmalı
  (`file app.py` → "with CRLF line terminators").
- Karar/iş mantığı arayüzde değil: kural kararlarını üreten işlev `signal_rules.py`'ye (ör.
  `rule_decisions(symbol, frame, v1_decisions) -> (decisions, method)`) eklenir; `app.py` yalnız çağırır.
- Haftalık ve diğer periyotlar kapsam dışı (bugünkü araştırma amaçlı backtest aynen).
- **Kapsam dışı:** portföy testi (0019–0021 zaten kuralları kullanıyor), sanal takip, yeni kural/parametre denemesi.

## Context
- İlgili modüller: `app.py` (≈ satır 657–760, "📊 Backtest" bloğu ve `perf_report_source`), `signal_rules.py` (0022),
  `strategy_candidates.py` (`asset_method`, `trend_candidate`, `channel_candidate`, `candidate_decisions`, `TUT`),
  `technical_analysis.py` (`build_v1_decisions`, `run_v1_strategy_backtest`), `performance_ui.py`,
  `protection_ui.py`, `candidate_ui.py`, `trade_diagnostics.py` (`first_tradable_pos`).
- Terimler: **C / TREND_SMA** = V1 AL + kapanış > 200 günlük basit ortalama, çıkış V1 SAT; **kanal / KANAL** =
  kapanış önceki 50 günün en yükseğinin üstünde AL, önceki 20 günün en düşüğünün altında SAT; **TUT** = stopsuz al-tut.
- İlgili spec'ler: 0016/0017 (teşhis), 0015 (al-tut karşılaştırması), 0019, 0020, 0021, 0022.
- Kanıt/deney kodu: `docs/evidence/strateji-deneyleri/`.

## Acceptance Criteria
- [ ] **AC01 — Kural kararları (R01):** `rule_decisions` BTC için `TREND_SMA` kararlarını (V1 AL + ortalama üstü), SOL için
  kanal kararlarını (yalnız ilk V1 karar gününden itibaren), döviz ve "V1" modunda V1 kararlarını aynen döndürür.
- [ ] **AC02 — Hisse (R01):** AAPL / THYAO.IS'de zamanlama backtest'i çalışmaz; ekranda "al-tut" bilgisi ve al-tut sonucu görünür.
- [ ] **AC03 — Etiket (R02):** Ekranda BTC günlük backtest'inde kural adı ("V1 + 200 günlük ortalama") yazar; "V1 (eski)" modunda eski V1 metni yazar.
- [ ] **AC04 — Paneller (R03/R04):** Backtest sonrası `perf_report_source["decisions"]` kural kararları, `["v1_decisions"]` V1 kararlarıdır; aday değerlendirmesi V1 kararlarını kullanır.
- [ ] **AC05 — Motor eşliği (R05):** Aynı karar sözlüğüyle `run_v1_strategy_backtest` ve `portfolio_backtest.run_portfolio` (%100 ağırlık) aynı giriş/çıkış günlerini üretir (kanal kararlarıyla; 0019 AC05'in kanal karşılığı).

## Definition of Done
- [ ] Tüm kabul kriterleri test/kanıtla karşılandı; her kriter bir pytest testi (`docs/testing.md`, docstring `ACnn —`).
- [ ] Arayüz: kritik akış smoke test (AppTest; günlük veri için `tests/test_signal_rules.py::_fresh` ve `_daily` kalıbı).
- [ ] Yeşil pipeline (`pytest`, `-m browser`); `-m perf` bozulmadı.
- [ ] Bağımsız QA kabulü (AGENTS.md: üreten denetlemez), squash-merge (`docs/git.md`).

## Devir notu (2026-10-11 durumu)
- **Açık PR:** [#37](https://github.com/yilmaz-hub/analiz_app/pull/37) (0022) bağımsız QA'da; QA KABUL + CI yeşil ise
  squash-merge edilecek. Bu spec ona bağlı.
- **Çalıştırma:** `pip install -r requirements-dev.txt`; `python -m pytest -q` (≈ 905 test, ~6 dk), `-m browser`, `-m perf`.
- **Ertelenen düşük öncelikli notlar:** "ML: Devre dışı" neden satırı metni; `paper_trading`/`optimize_weights`
  belge yorumları; teşhiste başa-baş işlemin "kaybeden" sayılması; `trade_diagnostics._regime_before` geniş
  `except`'ine günlük kaydı; GC=F kodu `market_of`'ta tanınmıyor (uygulamanın altını `XAU_GOLD`).
- **Kullanıcı tarafı:** varlık başına "Adet/lot adımı" girilmeli (ör. ETH 0.001); yoksa sanal takip işlem hesaplamaz.

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | — |
| Bulgu gerçek/gürültü oranı | — |
| Regresyon sayısı | — |
| Kaçan hata | — |
