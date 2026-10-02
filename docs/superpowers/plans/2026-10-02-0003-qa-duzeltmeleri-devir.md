# Devir notu — Spec 0003 QA düzeltmeleri (Q1–Q14)

> Dal: `feature/0003-qa-duzeltmeleri` (dalı `main` @ c735697'den açıldı).
> Rol: Developer. Karar yetkisi Takım Yöneticisinde; QA kod yazmaz.
> Bu not, oturum sınırı nedeniyle işin başka bir oturumda (ör. bulutta) sürmesi içindir.

## Başlamadan önce
1. `AGENTS.md` ve `docs/` kurallarını oku. Spec: `specs/0003-islem-kararlari-ve-performans-tutarliligi.md`.
2. `pip install -r requirements-dev.txt`, sonra `python -m pytest tests/ -q` (şu an **378 geçiyor**).
3. Commit biçimi: Conventional Commits + `[plan 0003/qa-N]`. Doğrudan `main`'e commit yok.
4. Terminal Windows ise Python çıktısı için `PYTHONIOENCODING=utf-8` kullan.
5. Çalışma ağacında başka oturuma ait commit'lenmemiş `docs/*` değişiklikleri olabilir; bunlara dokunma, `git add -A` kullanma, dosyaları adıyla ekle.

## Her düzeltme için kural
Düzeltmeyi geri alınca düşen bir test yaz (önce düşür, sonra düzelt). Düzeltme geri alınıp testin düştüğünü doğrula.

## Durum

| Bulgu | Durum |
|---|---|
| Q9 stop yükseltme güvenli yazma | **Bitti**, commit `e51197c` |
| Q6 bileşen hazırlığı | **Yarım.** `market_validation.component_gaps` + `validate_market_data(require_components=..., missing_components=...)` ve testleri hazır. `signal_engine`: `_score_ml/_score_patterns/_score_advanced` artık `(puan, gerekçe, hazır_mı)` döner; `_compute_bar_score` sonucu `"unavailable"` taşır; `generate_stable_signal(strict_components=True)` eksik bileşende `CompositeSignal.unavailable_components` ile döner; `generate_validated_signal` strict'i varsayılan açar. **Eksik:** (a) bu yeni sinyal motoru davranışı için test (`ml` None/istisna → `unavailable_components == ("ml",)`; nötr ML sonucu eksik sayılmaz; formasyon istisnası; `include_ml=False` iken ml eksik sayılmaz), (b) `build_v1_decisions` hesaplanamayan barda `"BILESEN_YOK"` yazsın (yeni AL/SAT yok), (c) `app.py` ve `paper_trading.run_paper_update` 7 sütunluk elle denetim yerine `validate_market_data(..., require_components=True)` kullansın ve `comp_sig.unavailable_components`'ı BILESEN_HAZIR_DEGIL + bileşen adlarıyla göstersin, paper'da pending'e AL/SAT yazmasın. |
| Q5, Q11, Q12 | Başlanmadı (tasarım aşağıda) |
| Q13 | Başlanmadı |
| Q1, Q7, Q8, Q2, Q4 | Başlanmadı (çekirdek; tasarım aşağıda) |
| Q3 | Başlanmadı |
| Q10, Q14 | Başlanmadı; Q1 bittikten sonra |

## Doğrulanan bulgular (kanıt)
Q1–Q14'ün hepsi `main`'de koddan doğrulandı. Q1 (geçmiş test pnl 100 / sanal takip 97,9), Q2 (açık pozisyon, stop delinmiş → panel "TUT"), Q3 (aynı teyit iki kez kabul), Q4 (zararlı çıkıştan sonra panel "SATIN AL"), Q8 (270 TL'lik varlıkta 3,70370370 adet) çalıştırılarak birebir üretildi. Ek tespit: `portfolio.check_active_positions_auto_close` stopu `SL` anahtarından okuyor, arayüz ise `Stop` yazıyor; arayüzün açtığı pozisyonlarda stop uyarısı hiç çıkmıyor (Q2'ye dahil et, `Stop` öncelikli, `SL` yedek).

## Tasarım kararları

**Q1 / Q7 / Q8 — ortak kurallar.** `trade_decisions.py` ve `trade_execution.py` ortak kütüphane; üç motorun hepsi buradan geçecek:
- `TradeSettings(notional, quantity_step, costs)` ve `CostAssumptions` (makas, kayma, komisyon; `None` = bilinmiyor, `0` = açıkça sıfır).
- `fill_price` makas/kaymaya, `commission_fee` komisyona bağlı; ikisi birbirinden bağımsız. Komisyon alış tutarına ek olarak nakitten düşülür, satışta çıkan tutara uygulanır.
- `enter_position` → `initial_stop` + `fill_price` + `execute_purchase`. `initial_stop` `None` dönerse alış yapılmaz (AC110). `execute_purchase`'ta `initial_stop` verilmediyse denetim atlanır, açıkça `None` verildiyse `GECERSIZ_STOP` döner (sentinel kullan).
- `resolve_exit`: önce açılış fazı `decide_action(stop_touched = açılış <= stop)`, sonra gün içi faz `decide_action(Signal.WAIT, stop_touched = low <= stop)`. Fiyat için `evaluate_stop`.
- `classify_exit` bekleme kararını verir; `LOSS_COOLDOWN_BARS = 2`.
- Gizli `BacktestConfig.FEE_RATE` sanal takipten kalkar. Komisyon bilinmiyorsa her iki motor brüt model + `net_verified=False` üretir; sonuç "doğrulanmış net" diye sunulmaz.
- `run_v1_strategy_backtest(df, decisions, *, initial_cash, trade_notional, quantity_step=None, costs=None)`: `quantity_step=None` ise alım yok ve neden `MIKTAR_ADIMI_BILINMIYOR` (AC102). Mevcut testler `quantity_step` verecek şekilde güncellenir.
- Kullanıcı varsayımları (sermaye, tutar, adım, makas, kayma, komisyon) tek "İşlem varsayımları" panelinde girilir, backtest ve sanal takip aynı değeri kullanır, kalıcı depoda saklanır (`storage` belgesi, 0004'ün dersi: yayında yeniden başlatmada silinmesin). Boş alan = bilinmiyor, `0` = açıkça sıfır; negatif reddedilir (AC26).
- Parite testleri üç **gerçek** motoru aynı veriyle çalıştırır: backtest vs `advance_pending_daily_decision` (işlemler, fiyatlar, net), panel eylemi vs `decide_action` eşlemesi.

**Q2 / Q4 — panel.** `PanelInput`'a `current_price` ve `bars_since_loss_exit` ekle. Eylem `decide_action(signal, position, stop_touched=current_price<=stop, cooldown_bars=...)` ile belirlenir; temas varsa sinyalden bağımsız "TAMAMINI SAT" + gerekçe mesajı. Fiyat yoksa "güncel risk değerlendirilemiyor" mesajı. `app.py`'deki satır içi `stop_default` da `initial_stop` kullanmalı.

**Q3 — teyit formu.** Alış ve satışta işlem zamanı (Europe/Istanbul, gelecek olamaz) ve miktar alanı eklenir; `event_id` işlemin kendi alanlarından (varlık, yön, miktar, fiyat, UTC işlem zamanı) türetilir. Önce portföy yazılır, sonra günlük; günlükte zaten varsa ama portföyde `JournalEventId` yoksa yarım kalan işlem tamamlanır, ikisinde de varsa gerçek tekrar olarak reddedilir. Bakiye ve pozisyon değişikliği doğrulamadan sonra yapılır.

**Q5 / Q11 / Q12.** Doğrulama başarısızsa geçerli "BEKLE" üretme: `CompositeSignal`'a `data_status` ekle; yan çubuk ve ana kart hatayı nedenine göre (VERI_YOK, GECERSIZ_VERI, YETERSIZ_GECMIS, BILESEN_HAZIR_DEGIL + bileşen adları, YENI_VERI_BEKLENIYOR, ESKI_VERI) ayrı gösterir. `decision_at` son geçerli kararın zamanıdır (`st.session_state` içinde sembol başına saklanır) ve "güncel değil" notuyla ekrana basılır. `trading_ui`'ye kod → Türkçe metin eşlemesi ekle; ekranda ham kod (POZISYON_BILINMIYOR vb.) görünmesin; bilinmeyen kod için güvenli genel metin.

**Q13.** `PositionJournal` kalıcı depoya yazar (`storage` belgesi). Açılışta yerel `position_journal.json` varsa bir kereliğine kopyalanır. Okuma hatasında boş günlükle devam edilmez (0004 dersi).

**Q10 / Q14.** `test_trading_parity` gerçek motorlara geçince `trading_service.py` ölür, sil. `calculate_trailing_stop`, `advance_v1_book`, `_step_book`, `apply_signal`, `ignore_signal`, `aggregate_totals` ölü kod, sil ve test bağımlılıklarını gerçek davranış testlerine çevir. AC93 testi spec'e aykırı (USD+USDT toplanıyor); fonksiyon silinince `paper_report` para birimlerini ayrı tutuyor mu testiyle karşılanır. AC40/AC77 uygulama düzeyinde test edilir (SAT sinyali gerçek pozisyonu kapatmaz; teyitsiz AL gerçek işlem eklemez). Uçtan uca smoke: teyitli alış → panel TUT → SAT sinyali → TAMAMINI SAT → teyitli satış → pozisyon yok.

## Kapanış
- Spec 0003 SCORECARD'ına kaçan hataları (Q1–Q14) işle.
- Her adımda tam paket yeşil; PR şablonu doldurulur; squash-merge Takım Yöneticisi onayıyla.
