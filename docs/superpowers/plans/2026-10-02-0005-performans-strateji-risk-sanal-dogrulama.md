# Spec 0005 — Uygulama Planı

> Developer şapkası; plan inceleme/onay içindir. **Kod ve test yazılmadı.**
> Spec: `specs/0005-performans-strateji-risk-ve-sanal-dogrulama.md`, Revizyon 1 — R01–R19, AC01–AC87.
> Durum: **Taslak.** Spec'in kendisi onaylı değil (Q01–Q08 açık); bu plan, Q önerilerinin olduğu gibi kabul edileceği varsayımıyla yazıldı. Bir Q farklı karar çıkarsa yalnız o Q'ya bağlı dilim yeniden planlanır.

**Goal:** Mevcut AL→SAT yaklaşımının güvenilir performans raporunu çıkarmak, aday stratejileri ön kayıtlı koşullarda karşılaştırmak, pozisyon riskini sınırlamak ve seçilen sürümü ekran kapalıyken ileri dönemde izlemek.

**Architecture:** Mevcut Python/Streamlit yapısı korunur ([conventions.md](../../conventions.md) modül haritası: arayüz → karar → analiz → veri). Hesap çekirdekleri saf (Streamlit'siz, `Decimal`) modüllerdir; kalıcılık `storage.py` belge deposundan; arayüz `trading_ui.py` desenindeki ayrı bir dosyadadır, `app.py` yalnız akışı kurar. Ekran kapalıyken takip, aynı modülleri çağıran başsız bir koşucudur (`python -m forward_runner`).

**Tech Stack:** Python 3.13, pandas, SQLAlchemy (`storage.py` zaten kullanıyor), Streamlit `AppTest`, pytest. Yeni bağımlılık yok.

## Dayanılan mevcut kod

| Var olan | 0005'te kullanımı |
|---|---|
| `trade_execution.py` (`evaluate_stop`, `execute_purchase`, `round_stop_up`) | Stop teması `active_at` ile, adıma aşağı yuvarlama; AC33/59/60 için yeniden kullanılır, çoğaltılmaz |
| `trade_decisions.py` (`decide_action`, `initial_stop`, `classify_exit`) | Referans strateji (V1) — **değiştirilmez**, R01/R09 |
| `trading_service.py` (`evaluate_all_modes`, `DecisionLedger`, `point_in_time_value`) | Gelecek verisi yok kuralı (AC12); aday değerlendirmesi aynı girişten geçer |
| `technical_analysis.run_v1_strategy_backtest` / `build_v1_decisions` | Referans ve adayların tek simülasyon girişi |
| `position_journal.py` | Gerçek teyitli işlemler (`executed_at` / `recorded_at` zaten ayrı → AC44) |
| `storage.py` (`read_doc`/`write_doc`) | Ayar, aday kaydı, risk ayarı, aktif strateji |
| `config.py` `RegimeConfig`, `DecisionEngineConfig.REGIME_MA_PERIOD` | Rejim sınıflandırıcısının girdisi (Q02) |

**Dikkat:** `paper_trading.paper_report()` `float` ve `round()` kullanıyor. Yeni rapor onu **sarmaz**; `Decimal` ile sıfırdan yazılır, eskisi dokunulmadan kalır (Altın Kural 2).

## Yeni / değişen dosyalar

| Yol | Katman | Sorumluluk |
|---|---|---|
| `performance_report.py` | karar | Sermaye dizisi → net getiri, maks. düşüş, beklenti, filtre, yok değeri, "üst sınır" etiketi |
| `evaluation_window.py` | karar | Ayar/değerlendirme bölmesi (aşağı yuvarla), al-tut referansı, eşit-koşul denetimi |
| `regime_classifier.py` | analiz | Yükselen/düşen/yatay sınıfı; sürümlü, sabit eşik |
| `strategy_candidates.py` | karar | Aday kaydı, parmak izi, ön kayıt, Q03 kriterleri, değerlendirme geçmişi, aktif strateji |
| `risk_sizing.py` | karar | Miktar formülü, toplam risk, kayıp sınırı, sıfırlama kaydı |
| `profit_protection.py` | karar | Sanal hedef / iz süren stop / kademeli çıkış adayları (gerçek pozisyona yazmaz) |
| `forward_tracker.py` | karar | Günlük kapanış kararı, tekillik, revizyon, kesinti, yeterlilik sayacı |
| `forward_runner.py` | arayüz-dışı giriş | Başsız koşucu (`__main__`); zamanlayıcı tetikler |
| `performance_ui.py` | arayüz | Rapor, aday, risk, ileri takip panelleri |
| `storage.py` | veri | `forward_decisions` tablosu (UNIQUE) — aşağıdaki karar D1 |
| `config.py` | ortak | `PerformanceConfig`, `RiskLimitConfig`, `ForwardConfig` (eşikler: 250, %60, 30, 90, 5 dk…) — sihirli sayı yok |
| `exceptions.py` | ortak | `ReportFilterError`, `RecordNotFoundError`, `RiskInputError` (`ProTraderError` alt sınıfları) |
| `app.py` | arayüz | Yalnız `performance_ui` çağrısı |
| `tests/test_*.py`, `tests/fixtures/` | — | Aşağıdaki dilimlere göre |

## Dilimler ve AC eşlemesi

Her dilim = bir spec aşaması (spec: "her aşama ayrı kabul kanıtı ve uygulama dilimi"). Sıra 4 → 5 → 6 → 7 sabittir. 87 AC'nin tamamı tam bir kez eşlenmiştir.

| Dilim | Aşama | AC | Bağlı karar |
|---|---|---|---|
| **S1** Rapor çekirdeği | 4 | 02–10, 49, 50, 55, 64–66, 78, 84 (17) | — |
| **S2** Dönem, al-tut, eşit karşılaştırma | 4 | 11–14, 47, 53, 54, 56, 57, 62 (10) | Q01 |
| **S3** Adaylar ve rejim | 5 | 01, 15–22, 48, 51, 52, 69, 79, 80, 85 (16) | Q02, Q03 |
| **S4** Risk büyüklüğü ve kayıp sınırı | 6 | 23–31, 58, 67, 68, 72–75, 81–83 (19) | Q04 |
| **S5** Kâr koruma adayları | 6 | 32–34, 59–61 (6) | Q05 |
| **S6** Sanal doğrulama (headless) | 7 | 35–44, 63, 70, 71, 76, 77, 87 (16) | Q06a, Q07, Q08 |
| **S7** Süre bütçeleri | 7 | 45, 46, 86 (3) | Q06b |

### Her dilimin işleyişi (TDD, [testing.md](../../testing.md))
1. O dilimin AC'leri için **önce kırmızı test**; her testin docstring'inin ilk satırı `ACnn` (denetim: `grep -rhoE "AC[0-9]{2}" tests/ | sort -u` ile eşleme sayılır).
2. Minimum kod → yeşil. Assert zayıflatma / skip / retry yok (AP-02/03).
3. Para/yüzde `Decimal("…")`; `quantize` tek noktada ve açık mod ile.
4. Arayüzü olan AC'ler için `AppTest` + ekran görüntüsü; boş/hata durumu dahil.
5. Dilim sonu: `python -m pytest tests/ -v` çıktısı + commit `feat(<modül>): … [plan 0005/<dilim>]`.

### S1 — Rapor çekirdeği
- `performance_report.py`: girdi = günlük kapanış sermaye dizisi + kapanmış işlem listesi + maliyet bilgisi. Çıktı: `ReportResult` (değerler `Decimal | None`; `None` = "hesaplanamıyor").
- Sermaye başlangıcı ≤ 0 → getiri `None` (AC49). Boş dönem → `None` (AC02, 78). Başlangıç = bitiş geçerli (AC55); başlangıç > bitiş `ReportFilterError` (AC09).
- Maks. düşüş yalnız günlük kapanış dizisinden (AC50). Açık pozisyon sermayeye girer, kapanmış sayıya girmez (AC06).
- Para birimi: sonuçlar `{para_birimi: Decimal}` sözlüğü; çapraz toplam yok (AC10).
- Eksik maliyet: `cost_missing_count` + `is_upper_bound` bayrağı; işlem örneklemden düşmez (AC07, 84).
- Filtre: piyasa/varlık/para birimi/dönem/sürüm/rejim; bilinmeyen değer `ReportFilterError`, bilinen-ama-boş değer örnek sayısı 0 (AC64, 65).
- **Arayüz sınırı notu:** AC64/66 "400/404" diyor; bu uygulamada HTTP API yok. Plan: `ReportFilterError` / `RecordNotFoundError` istisnaları + arayüzde anlaşılır mesaj (teknik metin sızmaz). **Spec ifadesi için onay istenir (AP-10, açık nokta O1).**

### S2 — Dönem ve karşılaştırma
- `evaluation_window.split(dates)`: `ayar = floor(n × 0,60)`, kalan değerlendirme (AC13, 53); n < 250 → `INSUFFICIENT_HISTORY` (AC54, 57).
- Al-tut: ilk uygun açılışta `execute_purchase` ile (adım + maliyet); kalıntı nakit dönem sonu sermayesine eklenir (AC47, 62).
- Karşılaştırma ancak başlangıç sermayesi ve dönem **tam eşitse** (Decimal eşitliği; 10.000 ≠ 10.000,01) (AC11); ortak tarih yoksa reddedilir (AC56); dış nakit hareketi → karşılaştırma uygun değil (AC14).
- AC12: karar tarihinden sonraki fiyatlar değiştirilince karar aynı → `trading_service.point_in_time_value` üzerinden test.

### S3 — Adaylar ve rejim
- `regime_classifier.py`: skor (`calculate_regime_score`) + MA konumundan 3 sınıf; eşikler `config.py`'de, çıktı `(sınıf, sürüm)` (AC85). Eşik değerleri **Q02 karar kaydında** sabitlenmeden yazılmaz (açık nokta O2).
- `strategy_candidates.py`: aday = {id, giriş, çıkış, piyasa koşulu, ayarlar, ölçütler, değerlendirme dönemi}; parmak izi = kanonik JSON'un SHA-256'sı (AC79). Ön kayıt yoksa koşum başlamaz (AC69); onaysız kural → "deneme" etiketi (AC48). Başarısız aday geçmişte kalır (AC16). Kırılım / geri çekilme ayrı kayıt (AC80).
- Q03 kriteri: `getiri > ref` ∧ `maks_düşüş ≤ ref` ∧ `beklenti > 0` ∧ `n ≥ 30`; sonuç üç değerli: `YETERSIZ_VERI` / `OLCUTU_KARSILAMADI` / `OLCUTU_KARSILADI` (AC18–22, 51, 52). Piyasa başına ayrı sonuç (AC22).
- Aktif strateji `storage` belgesi; varsayılan = mevcut V1; aday ölçütleri sağlansa da değişmez (AC01, 17).
- Filtre-etki karşılaştırması yalnız bir ayarı değişen çiftlerde "tek filtre" etiketi alır (AC15).

### S4 — Risk
- `risk_sizing.size(...)`: `floor((risk_bütçesi) / (giriş − stop + birim_maliyet), adım)`; yukarı yuvarlama yok. Geçersiz stop (≥ giriş, ≤ 0), bilinmeyen adım, eksik sermaye, tanımsız profil → miktar yok + neden (AC23–25, 31, 67, 68). Adım/asgari tutar altı → miktar yok (AC83).
- Oran doğrulama: `0 < r < 1`; yüzde ölçeğinde sıfıra yuvarlanan girdi reddedilir (AC26, 58). Ayar `storage` belgesinde, kalıcı (AC81).
- Toplam risk: aynı para biriminde; `max(0, giriş−stop)×miktar`; limit **dahil** (200+100 ≤ 300 geçer, 100,01 engellenir) (AC27, 28, 75). Karma para birimi → "doğrulanmadı" (AC73, 74).
- Kayıp sınırı: eşitlik engeller (AC29); sınır doluyken açık pozisyonun SAT/stop bilgisi görünür (AC30); yalnız kullanıcı eylemiyle sıfırlanır, eylem kaydı + yeni dönem başlangıcı yazılır (AC72).
- **Miktar adımı kaynağı:** bugün kodda varlık-başı adım tablosu görünmüyor (`execute_purchase` parametre alıyor). Açık nokta O3.

### S5 — Kâr koruma adayları
- `profit_protection.py`: sabit hedef / yalnız yukarı taşınan iz süren stop / kademeli çıkış, **ayrı** adaylar; parametreler Q05 ile sabitlenir. Gerçek pozisyon kaydı (`position_journal`) **salt okunur** geçilir (AC32).
- Stop teması `trade_execution.evaluate_stop(active_at=…)`; `≥` sınırı AC33/59 için önce test (kod muhtemelen zaten sağlıyor — kırmızı çıkmazsa yalnız regresyon testi olarak kalır).
- Kademeli çıkış: kapanan/kalan miktar ayrı, ürün adımına uygun (AC34, 60); son parça kapanınca kapanmış işlem +1 (AC61).

### S6 — Sanal doğrulama
- `forward_tracker.process(asset, version, candle_ts)`: karar kaydı = {kaynak, değerlendirme zamanı (UTC), dayanak mum zamanı, sürüm, varsayımlar, `on_time` bayrağı} (AC36, 38).
- Tekillik: `forward_decisions` tablosunda `UNIQUE(asset, strategy_version, candle_ts)`; ikinci yazım `IntegrityError` yakalanıp **yok sayılır** (AC37, 70). Mum revizyonu karar değiştirmez, `revision` kaydı ayrı yazılır (AC71).
- Yeterlilik sayacı yalnız `on_time = True` ve gerçek saatli kayıtları sayar; geç kayıt (Q08) ve hızlandırılmış/ileri alınmış saat sayacı etkilemez (AC63, 76, 87). Eşikler: 90 izlenen gün (89 yetmez), 30 işlem, 3 rejim (AC39–41); sürüm değişimi sayacı sıfırlar (AC42). Açık gerçek pozisyon giriş sürümüyle yönetilir (AC77).
- Sanal ve gerçek işlem ayrı koleksiyon/sayaç (AC43); geç kayıt `executed_at` ≠ `recorded_at` korunur (AC44 — `position_journal` zaten ayırıyor; regresyon testi).
- `forward_runner.py`: `python -m forward_runner`; `st.*` import etmez. Tetikleyici ve ortam Q06a ile onaylanır. AC35 testi: süreç tabanlı (alt süreç, sahte sağlayıcı, sabit saat) — "5 dakika" bütçesi koşucu içi hesap süresiyle ölçülür, gerçek zamanlayıcı gecikmesi ayrıca belgelenir.

### S7 — Süre bütçeleri
- `tests/fixtures/`: sabit 10.000 işlem kaydı ve 20×1.000×3 değerlendirme verisi (üretici betik + sabit tohum).
- `tests/test_performance_budgets.py`: ilk (soğuk) ve sonraki çalışma ayrı; 10 tekrar, **en kötüsü** bütçe altında (2 sn / 120 sn); sağlayıcı bekleyişi ayrı alan (AC45, 46, 86). Ortam (CPU/RAM/Python) rapora yazılır. Bu testler yavaş olduğundan `@pytest.mark.perf` ile ayrılır, CI'da ayrı adım.

## Önerilen branch / PR düzeni
`git.md` kuralına uygun, aşama başına bir branch + squash-merge, sıra bağımlı:

| Branch | Dilimler |
|---|---|
| `feature/0005-adim4-rapor` | S1, S2 |
| `feature/0005-adim5-adaylar` | S3 |
| `feature/0005-adim6-risk` | S4, S5 |
| `feature/0005-adim7-sanal-takip` | S6, S7 |

Her PR öncesi **bağımsız QA (B) oturumu**; ben kendi işimi denetlemem. Yeşil `pytest` + ekran görüntüleri PR'a eklenir; commit mesajını ben yazarım, yönetici onaylar.

## Kararlar (plan içi) — öneri + gerekçe

| # | Karar | Öneri | Gerekçe |
|---|---|---|---|
| D1 | Tekillik nerede zorlanır? | `storage.py`'ye ayrı `forward_decisions` tablosu, `UNIQUE(asset, strategy_version, candle_ts)` | `app_records` tek-belge deposu; "depoda zorlanır" (R17) ve eşzamanlı ikinci yazımın yok sayılması (AC70) için belge deposu yetmez. Aynı SQLAlchemy motoru, yeni bağımlılık yok. |
| D2 | Rapor `paper_report`'u genişletsin mi? | Hayır, yeni `Decimal` modül | Mevcut `float`/`round`; Altın Kural 2. |
| D3 | Aşama başına branch | Evet (tablo) | Spec "her aşama ayrı kabul kanıtı" der; küçük, QA'lanabilir PR'lar. |
| D4 | Yeni AC → test izlenebilirliği | Docstring'de `ACnn` | Basit, `grep` ile 87/87 doğrulanır. |

## Açık noktalar — yönetici kararı gerekir
- **O1 (AP-10):** AC64/AC66 "400/404" HTTP kodu söylüyor, uygulamada API yok. Öneri: domain istisnası + arayüz mesajı olarak okunsun; spec metni buna göre düzeltilsin (**AP-08**, onayınızla).
- **O2:** Spec "kripto için sabitlenmiş UTC kesim saati" diyor ama değeri yazmıyor; ayrıca rejim eşikleri (Q02) ve kâr koruma parametreleri (Q05) sayısal değil. Öneri: kesim 00:00 UTC (günlük mum kapanışıyla aynı); eşik/parametre değerleri S3/S5 başında ayrı karar kaydı olarak gelsin, onaysız kod yazılmaz.
- **O3:** Varlık-başı miktar adımı/asgari işlem tutarı kaynağı belirsiz. Öneri: varlık kaydında (`assets`) opsiyonel alan; boşsa AC31 gereği miktar önerilmez.
- **O4:** Q06a — başsız koşucunun çalışacağı yer (yerel makine Windows Görev Zamanlayıcısı mı, barındırma ortamı zamanlayıcısı mı?). Yerel SQLite ile yayın veritabanı farklı olduğundan koşucu ile uygulama **aynı** veritabanını görmeli. Spec: kurulum/işletim maliyeti plan aşamasında onaylanır → bu madde onay bekliyor; S6 buna kadar başlamaz.
- **O5:** 0003'ün kapanış durumu (dosya `specs/` kökünde, merge edilmiş görünüyor). Bağımlılık gereği QA doğrulaması gerekir.

## Riskler
- Uzun yaşayan koşucu + Streamlit süreci aynı SQLite dosyasında yazarsa kilitlenme olabilir → S6'da yazma yeniden denemesi değil, kısa işlem + testle kanıt.
- 20 varlık × 1.000 mum × 3 aday 120 sn bütçesi: `run_v1_strategy_backtest` hızı ölçülmeden söz verilmez; S7'de ölçüm, gerekirse S3'te vektörleştirme önerisi (plan dışı → AP-07).
- `float` ara hesap istisnası yalnız gösterge/ATR içindir; rapor/risk/sermaye hesabına sızmamalı (her dilimde `float(` taraması).

## Onay noktaları
1. Bu plan ve D1–D4, O1–O5 kararları → **Takım Yöneticisi**.
2. Spec onayı (Q01–Q08) → plan uygulamasının ön koşulu.
3. Her aşama sonunda QA oturumu + yönetici checkpoint'i; atlanmaz.
