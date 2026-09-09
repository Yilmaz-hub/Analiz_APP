# Spec: 0001 — Mobil Görünüm Modu (Grafiğin Ekrana Stabil Sığması)

> Bu şablondan üretildi: `specs/TEMPLATE.md`. Bitince → `specs/done/`.
> Kural: **spec'i olmayan feature'a başlanmaz** (bkz. `AGENTS.md`).
> **Revizyon 1 (CLARIFY):** Bu spec'in ilk hâli "katmanları gizleyerek sadeleştirme"
> üzerine kuruluydu. O yaklaşım **iptal edildi**. Yeni karar: mobilde hiçbir katman veya
> özellik gizlenmez; sorun grafiğin ekrana sığmamasıdır ve **görüntüyü stabilleştirerek**
> çözülür.

## Intent
Mobil kullanıcı siteyi telefondan açtığında analiz grafiği ekrana sığmıyor: grafik sabit
bir yükseklikle çizildiği için telefon ekranını taşıyor, alt kısmı kesik/yarım kalıyor ve
kullanıcı fiyatı bütün olarak görebilmek için sayfayı kaydırmak zorunda kalıyor. Bu
iyileştirmeyi, telefondan bakıp hızlı karar vermek isteyen son kullanıcı (trader) istiyor;
çünkü ekrana sığmayan, kesik görünen grafik aracın temel işini — fiyatı ve sinyalleri tek
bakışta görmeyi — engelliyor. Başarı, kullanıcının bir butonla Mobil görünüme geçtiğinde
grafiğin telefon ekranına stabil biçimde oturması, hiçbir yerinin kesik kalmaması ve bunu
yaparken masaüstü görünümündeki **hiçbir özelliğin kısılmamasıdır**. CLARIFY kararları:
(1) yığın Streamlit + Plotly'dir, React değil; (2) ekran genişliği otomatik algılanmaz —
kullanıcı Masaüstü/Mobil görünümü **butonla** seçer; (3) mobil görünümde masaüstüne göre
eksik veya gizli hiçbir katman, gösterge ya da özellik olmaz — çözüm özellik gizlemek
değil, görüntüyü stabilleştirmektir. Bilerek yapılmayan; tahmin motorunun hesap mantığını
değiştirmek, yeni gösterge (volume/RSI) eklemek veya masaüstü görünümünü değiştirmektir.

## Requirements
- Grafiğin **hemen üstünde**, Masaüstü ↔ Mobil görünüm arasında geçiş yapan tek bir kontrol
  bulunmalı. (Kenar çubuğuna konmaz: kenar çubuğu telefonda kapalı gelir, kontrol
  bulunamaz.)
- **Mobil görünümde grafik ekrana stabil oturmalı:** grafik yüksekliği masaüstündeki sabit
  `900px` yerine **`560px`** (ayarlanabilir varsayılan) değerine iner; grafiğin altı kesik
  kalmaz, kullanıcı grafiği bütün olarak görmek için kaydırmak zorunda kalmaz.
- **İlk açılışta varsayılan mod Masaüstü'dür.** Mobil görünüme kullanıcı butonla geçer.
- **Mobil modda başlangıç zoom penceresi daralır:** dar ekranda mumlar okunabilir kalsın
  diye başlangıçta gösterilen mum sayısı masaüstündekinin yarısıdır
  (`1wk`: 50→25, `1d`: 80→40, diğer: 100→50). Bu **veri gizlemek değildir** — tüm veri
  yerindedir, kullanıcı pan/zoom ile tamamına erişir; yalnızca açılış penceresidir.
- **Özellik eşitliği (bu spec'in çekirdek kuralı):** Mobil görünümde masaüstüne göre eksik
  veya gizli **hiçbir şey olmaz**. EMA 20, EMA 50 / bulut, AI Tahmini, AI Güven Skoru
  banner'ı, AI Trend çizgileri, W/M kutuları, mum ikonları, gelişmiş formasyonlar
  (üçgen / harmonik / Baş-Omuz), boyun çizgileri, HEDEF çizgileri ve mevcut fiyat çizgisi —
  hepsi iki görünümde de aynı şekilde çizilir.
- **Etkileşim eşitliği:** Pan, zoom, imleç okuması (hover) ve eksen rozetleri mobil
  görünümde de çalışır; hiçbir kontrol kaldırılmaz.
- Seçilen görünüm modu **oturum içinde korunur**; her yeniden çizimde (rerun) varsayılana
  dönmez. Kullanıcı seçimi, varsayılanı ezer.
- Görünüm modu değişimi fiyat, tahmin ve gösterge **değerlerini değiştirmemeli**; yalnızca
  yerleşimi (boyut/ölçek) etkiler.
- **Masaüstü görünümü** bu değişiklikten önce ne gösteriyorsa birebir aynısını göstermeli.
- Grafiğe ait hiç veri yoksa, kesik/boş eksen yerine sade bir **"veri yok"** durumu
  gösterilmeli.
- Veri alınamadığında kullanıcıya **teknik hata metni sızmamalı**; anlaşılır bir hata
  mesajı gösterilir, teknik ayrıntı log'a yazılır.

## Constraints
- **Yığın:** Streamlit `1.61` + Plotly `6.9` (Python). Grafik sunucu tarafında
  `ui_components.py` içinde üretilir. React/Vite/TypeScript **yoktur**; `docs/frontend.md`
  ve `docs/testing.md` bu projeyle çelişir (ayrı iş olarak düzeltilecek).
- **Ekran genişliği ölçülmez.** Streamlit sunucu tarafında tarayıcı viewport genişliğini
  bilmez; `< 768px` gibi bir eşik **kullanılmaz**. Mod yalnızca kullanıcının butonuyla
  belirlenir.
- **Hedef cihaz:** 6.7 inç telefon, dikey (portrait). CSS viewport ≈ `412×892`
  (Pixel 8 Pro / Galaxy S24+) — `430×932` (iPhone 15 Pro Max). Tarayıcı çubukları düşünce
  görünür yükseklik ≈ `750px`. `560px` grafik yüksekliği bu bütçeden türetilmiştir:
  `750 − 45` (Streamlit başlığı) `− 20` (sayfa boşluğu) `− 50` (mod butonu) `− 65`
  (grafiğin üstündeki AI Güven Skoru banner'ı) `− 20` (alt boşluk) `≈ 550`.
- **Bilinen sınır:** Yatay (landscape) kullanımda görünür yükseklik çok daha azdır; V1 bu
  durumu ayrıca ele almaz (bkz. Kapsam dışı).
- `st.set_page_config` uygulama başında yalnızca bir kez çalışır; bu nedenle sayfa düzeni
  (`layout`) ve kenar çubuğu başlangıç durumu buton ile değiştirilemez — mod yalnızca
  grafiğin kendi yerleşimini etkiler.
- **Performans:** Mobil görünüm masaüstüyle aynı sayıda katman çizdiği için render maliyeti
  değişmez; ölçülebilir kriter, çizilen öğe sayısının iki modda **eşit** olmasıdır
  (bkz. Acceptance Criteria).
- Para/yüzde gösterimi mevcut kurala uyar; bu spec değer hesabına dokunmaz.
- **Yasak:** Tahmin/analiz hesaplama mantığını değiştirmek; yeni gösterge/panel eklemek;
  masaüstü görünümünü değiştirmek; mobilde herhangi bir katmanı gizlemek/kısmak.
- **Kapsam dışı (V1):**
  - Katmanları gizleme / varsayılan olarak kapatma (bu spec'in iptal edilen ilk yaklaşımı).
  - Ekran genişliğinin otomatik algılanması ve moda otomatik geçiş.
  - Görünüm tercihinin **oturumlar arası** hatırlanması.
  - Volume ve RSI alt panelleri (ayrı spec).
  - Yeni göstergeler, yeni zaman aralığı kontrolleri veya çizim araçları.
  - Grafik dışındaki sayfa bölümlerinin (risk paneli, tablolar, kenar çubuğu) yerleşimi.
  - Yatay (landscape) yönelim için ayrı bir yerleşim.
  - Grafik kütüphanesinin veya tahmin motorunun değiştirilmesi.

## Context
- İlgili modül(ler): `ui_components.py` → `render_main_chart` (grafik üretimi ve
  `update_layout` yerleşimi), `app.py` (katman anahtarları ve çağrı), `theme.py`
  (CSS enjeksiyonu, eksen rozetleri) — bkz. `docs/architecture.md`.
- İlgili alan terimleri: AI Tahmini, Güven Skoru, EMA, direnç/destek (Dir/Dst), HEDEF —
  bkz. `docs/domain.md`.
- Referans görseller: mevcut mobil görünüm (`stream.jpeg`), hedeflenen mobil düzen
  (`trading.jpeg`).
- Mevcut durum: grafik `height=900` sabitiyle çiziliyor; genişlik
  `use_container_width=True` ile zaten uyum sağlıyor. Sorun **yükseklik**tedir.
- Mevcut başlangıç zoom penceresi zaman aralığına göre `50 / 80 / 100` mumdur; dar ekranda
  100 mum ≈ 3.6 px/mum ile okunamaz hâle gelir, yarıya inince ≈ 7 px/mum olur.
- Yeni sabitler `config.py` altında tutulur (grafik yüksekliği ve mobil mum sayıları
  cihaz üzerinde görüldükten sonra ayarlanabilsin diye).
- Bağımlılıklar / ilgili spec'ler:
  `docs/superpowers/specs/2026-08-06-chart-ux-improvements-design.md`.

## Acceptance Criteria
> Her satır **tek başına test edilebilir** olmalı (bkz. `docs/testing.md`).
- [ ] **Mod kontrolü:** Grafiğin üstünde Masaüstü/Mobil kontrolü görünür ve tıklandığında
      diğer moda geçer.
- [ ] **Varsayılan mod:** Uygulama ilk açıldığında mod Masaüstü'dür.
- [ ] **Mobil yükseklik:** Mobil modda üretilen figürün yüksekliği `560`'tır.
- [ ] **Mobil zoom penceresi:** Mobil modda başlangıçta görünen mum sayısı, aynı zaman
      aralığındaki masaüstü değerinin yarısıdır (`1wk`: 25, `1d`: 40, diğer: 50).
- [ ] **Zoom penceresi veri gizlemez:** Mobil modda figüre eklenen veri noktası sayısı
      masaüstüyle aynıdır; yalnızca `xaxis.range` başlangıç değeri farklıdır.
- [ ] **Masaüstü regresyonu yok:** Masaüstü modda üretilen figürün yüksekliği `900`'dür ve
      yerleşim (margin, eksen tarafı, hover modu, dragmode) değişmemiştir.
- [ ] **Özellik eşitliği (çekirdek):** Aynı sembol, zaman aralığı ve aynı katman
      anahtarlarıyla, mobil modda üretilen figürdeki **trace + shape + annotation sayısı**,
      masaüstü modda üretilenle **birebir eşittir**.
- [ ] **Etkileşim eşitliği:** Pan/zoom ve hover ayarları (`dragmode`, `scrollZoom`,
      `hovermode`, modebar) iki modda da aynıdır.
- [ ] **Oturum kalıcılığı:** Mobil moda geçildikten sonra herhangi bir başka kontrol
      kullanıldığında (rerun) mod Mobil kalır, varsayılana dönmez.
- [ ] **Değer koruması:** Mod değişimi fiyat serisini ve tahmin değerlerini değiştirmez.
- [ ] **Boş durum:** Grafiğe ait hiç veri yokken grafik çizilmez; sade bir "veri yok"
      durumu gösterilir.
- [ ] **Hata durumu:** Veri alınamadığında kullanıcıya anlaşılır bir mesaj gösterilir;
      ham hata/stack trace ekrana yazılmaz.

## Definition of Done
- [ ] Tüm kabul kriterleri test/kanıtla karşılandı.
- [ ] Testler: her kriter için `pytest` testi (`tests/test_chart_layout.py` genişletilir).
      Figür nesnesi üzerinden doğrulanır — yükseklik, öğe sayısı, yerleşim ayarları.
      *(Not: `docs/testing.md` "xUnit" diyor; bu proje Python'dur, `pytest` geçerlidir.)*
- [ ] Arayüz kanıtı: ekran görüntüleri — mobil mod (grafik tam sığmış), masaüstü mod
      (değişmemiş), boş durum, hata durumu.
- [ ] Kritik akış smoke test: mod değiştir → başka bir kontrole dokun → modun korunduğunu
      doğrula.
- [ ] Yeşil pipeline (lint + test).
- [ ] Para/yüzde alanları decimal; kullanıcıya teknik hata sızmıyor.
- [ ] PR şablonu dolduruldu, squash-merge (`docs/git.md`).

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 2 |
| Bulgu gerçek/gürültü oranı | 8/0 |
| Regresyon sayısı | 1 |
| Kaçan hata | - |

> Notlar: **Düzeltme turu (2):** QA 1. tur (8 bulgu) + QA 2. tur (yeniden açılan
> BULGU-5). **Bulgu 8/0:** sekiz bulgunun tamamı git/ölçüm ile doğrulandı, gürültü yok.
> **Regresyon (1):** BULGU-4 — `main`'deki `st.error("Veri Alınamadı")` sinyali
> geçici olarak sessiz `st.info`'ya düşmüştü; boş/hata ayrımıyla giderildi.
> **Kaçan hata:** henüz üretimde gözlenmedi (-).
