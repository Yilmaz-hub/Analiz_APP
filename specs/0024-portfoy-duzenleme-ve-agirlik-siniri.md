# Spec: 0024 — Portföy Düzenleme ve Ağırlık Sınırı

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 3 (2026-10-12).
> Durum: **ONAYLANDI — GELİŞTİRİLECEK.** Kaynak: Takım Yöneticisi onayı (2026-10-11 / 2026-10-12). Kararlar: Varsayılan ağırlıklar toplamı %100 (BTC 20, ETH 20, XAU 15, SOL/AVAX/XRP/LINK 10, DOGE 5; varsayılan listede LINK yer alır); toplam sınır kuralı 0 < toplam ≤ %100 (kalan nakitte); ara girdilerde form (`st.form`) ile yenilenmeme; motor seviyesinde `ValueError`.

## Intent
Kullanıcı portföy geçmiş testi ekranında varlık ağırlıklarını ve parametrelerini belirlerken, her bir girdi değişiminde sayfanın baştan yenilenmesi düzenleme sürecini aşırı yavaşlatmakta ve kullanıcı deneyimini bozmaktadır. Ayrıca portföydeki varlıkların ağırlık toplamı %100'ü aştığında (ör. %110) veya geçersiz olduğunda sistem hiçbir hata vermeden testi çalıştırmakta, bu da matematiksel ve finansal olarak gerçek dışı (aşırı kaldıraçlı/çelişkili) test sonuçlarına yol açmaktadır. Başarı: Kullanıcının tüm portföy ağırlıklarını ve ayarlarını sayfa donmadan/yenilenmeden bir form (`st.form`) içinde kesintisiz düzenleyebilmesi; toplam ağırlık %100'ü aştığında veya 0 olduğunda testin açık bir Türkçe uyarıyla engellenmesi; toplam ağırlığın ekranda ve motor seviyesinde güvenle doğrulanmasıdır.

## Requirements
- **R01 — Kesintisiz Parametre Düzenleme (Form Tabanlı Toplu Uygulama):** Varlık seçimi (`st.multiselect`) dinamik girdi kutularını yönetebilmek için formun dışında tutulmalıdır. Varlık ağırlıkları, başlangıç sermayesi, komisyon ve strateji onay kutuları bir `st.form` içine alınmalıdır. Kullanıcı bu parametreleri düzenlerken ara girdilerde sayfa yeniden yüklenmemeli (rerun olmamalı); parametreler ancak kullanıcı formu gönderdiğinde ("📦 Portföy testini başlat" submit butonu) tek seferde işletilmelidir.
- **R02 — Toplam Ağırlık Doğrulaması ve Sınırlar:** Tekil ağırlıklar `0 <= weight <= 100` aralığında olmalıdır. Seçilen varlıkların ağırlıkları toplamı `0 < toplam <= 100` (`Decimal("100")`) aralığında olmalıdır. Toplam ağırlık %100'ü aştığında veya toplam ağırlık 0 olduğunda testin başlatılması engellenmeli; kullanıcıya mevcut toplam ağırlık ve aşım miktarı açıkça bildirilmelidir.
- **R03 — Toplam Ağırlık ve Boşta Nakit Gösterimi:** Kullanıcı testi başlattığında geçerli ağırlıklar için güncel ağırlık toplamı ve %100'den az ise kalan boş nakit oranı (ör. "Toplam Ağırlık: %80, Boşta Nakit: %20") ekranda bilgilendirme özeti olarak gösterilmelidir.
- **R04 — Motor Seviyesinde Doğrulama (Sıfır Sessiz Hata):** Portföy hesap motoru (`portfolio_backtest.run_portfolio`), toplam ağırlığın %100'ü aştığı veya 0 olduğu girdileri kabul etmemeli ve açık bir doğrulama hatası (`ValueError`) ile reddetmelidir. Eski `_hold` içindeki sessiz oran küçültme (`scale = 100 / total`) kodu ve docstring temizlenmeli; sessizce oran küçültme veya aşırı ağırlıkla çalıştırma yapılmamalıdır.
- **R05 — Varsayılan Ağırlıklar ve Liste Uyumu:** Varsayılan listede BTC (%20), ETH (%20), ONS ALTIN (%15), SOL (%10), AVAX (%10), XRP (%10), LINK (%10) ve DOGE (%5) yer almalı ve toplamları tam olarak %100 (`Decimal("100")`) olmalıdır. `DEFAULT_COIN_MAP` listesinde LINK ("Chainlink (LINK)": "LINK-USD") yer almalı; `portfolio_backtest.default_weight("XAU_GOLD")` değeri `Decimal("15")` olarak güncellenmelidir.

## Constraints
- Para ve yüzde hesaplamaları her zaman `Decimal` tipindedir (`docs/conventions.md`). Float yasaktır.
- İş ve doğrulama mantığı arayüzle sınırlı kalmaz; hesap motoru (`portfolio_backtest.py`) bağımsız olarak ağırlık sınırını denetler.
- Streamlit mimari kuralı: `st.multiselect` form dışında, parametre girişleri ve çalıştırma butonu `st.form` içinde yer alır.
- `app.py` satır sonları CRLF olarak korunmalıdır.
- **Kapsam dışı:** Portföy optimizasyonu/otomatik ağırlık dağıtıcı algoritması, canlı emir gönderimi, negatif ağırlık (kısa pozisyon).

## Context
- İlgili modüller: `portfolio_ui.py` (ekran paneli ve form düzeni), `portfolio_backtest.py` (motor, ağırlık denetimi, XAU varsayılan ağırlığı), `config.py` (`DEFAULT_COIN_MAP` LINK tanımı), `tests/test_portfolio_weights.py` (yeni test dosyası), `tests/test_trend_portfolio.py` (XAU test iddiası güncellemesi).
- Terimler:
  - **Portföy Ağırlığı:** Her bir varlığa toplam sermayeden ayrılan yüzde pay (`Decimal`). Tekil aralık [0, 100].
  - **Toplam Ağırlık:** Portföydeki tüm varlıkların ağırlıklarının toplamı (`sum(weights)`). Geçerli aralık: 0 < toplam ≤ %100.
  - **Boşta Nakit Payı:** %100 ile toplam ağırlık arasındaki fark. İşleme girmeyen rezerv nakit.
- İlgili spec'ler: 0019 (ağırlıklı portföy), 0020 (kanal kırılımı), 0021 (al-tut ve döviz).

## Acceptance Criteria
- [x] **AC01 — Düzenleme Sırasında Yenilenmeme (R01):** Portföy ağırlıkları ve parametre kutuları değiştirilirken sayfa ara çalıştırma (rerun) yapmaz; test yalnızca form submit butonuna tıklandığında çalışır (`st.form`).
- [x] **AC02 — %100 Aşımında ve Sıfır Toplamda Engelleme (R02, R03):** Ağırlıklar toplamı %100'ü aştığında (ör. %110) ekranda aşım miktarıyla birlikte uyarı verilir ve test başlatılamaz; toplam ağırlık 0 olduğunda da uyarı verilir ve test çalıştırılmaz.
- [x] **AC03 — Geçerli Ağırlıklarla Başlatma ve Nakit Payı (R02, R03):** Ağırlıklar toplamı ≤ %100 olduğunda (ör. %100 veya %80) test başarıyla çalışır; %100'ün altında ise ekranda kalan nakit payı gösterilir ve motor seviyesinde kalan payın nakitte kaldığı doğrulanır.
- [x] **AC04 — Motor Seviyesinde Hata ve Sessiz Ölçekleme Temizliği (R04):** `portfolio_backtest.run_portfolio` fonksiyonuna toplamı %100'ü aşan veya 0 olan ağırlıklar verildiğinde `ValueError` fırlatılır; `_hold` içindeki eski oran küçültme kodu kaldırılmıştır.
- [x] **AC05 — Varsayılan Portföy Geçerliliği ve XAU %15 (R05):** Varsayılan varlıklar (BTC %20, ETH %20, XAU %15, SOL %10, AVAX %10, XRP %10, LINK %10, DOGE %5) ile ekran ilk yüklendiğinde toplam ağırlık %100'dür ve test çalıştırılmaya hazırdır. `default_weight("XAU_GOLD") == Decimal("15")` olarak doğrulanır.

## Definition of Done
- [x] Tüm kabul kriterleri test/kanıtla karşılandı.
- [x] Yeni testler `tests/test_portfolio_weights.py` dosyasında yazıldı; `tests/test_trend_portfolio.py` içindeki XAU iddiası güncellendi (`docs/testing.md`).
- [x] Arayüz smoke testleri (`AppTest`) ile form davranışının (rerun olmaması) ve hata uyarılarının çalıştığı kanıtlandı.
- [x] Yeşil test pipeline (`pytest`).
- [x] Para/yüzde alanları `Decimal`; kullanıcıya teknik hata / traceback sızmıyor.
- [x] Bağımsız QA kabulü ve PR süreci.

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 3 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | 6/0 |
| Regresyon sayısı | 0 |
| Kaçan hata | — |
