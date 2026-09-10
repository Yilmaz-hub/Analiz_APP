# Spec: 0002 — Mobil Grafik Yatay Doldurma

> Bu şablondan üretildi: `specs/TEMPLATE.md`. Bitince → `specs/done/`.
> Kural: **spec'i olmayan feature'a başlanmaz** (bkz. `AGENTS.md`).
> İlgili: spec 0001 (mobil görünüm modu — yükseklik/zoom). Bu spec onun bıraktığı
> **yatay** sorunu çözer.
>
> **Revizyon 1 (kök neden düzeltmesi):** İlk uygulamada darboğazın, x-ekseninin AI
> tahmini için geleceğe uzatılması olduğu varsayıldı ve mobilde bu uzatma kaldırıldı.
> Cihazda doğrulandığında görüntü **değişmedi**. Ekran görüntüsünün piksel ölçümü asıl
> nedeni gösterdi: çizim alanı ekranın yalnızca ~%31'i (x≈60–345/920), legend ise
> çizim alanının **sağ dışında** (x≈355–860). Plotly'nin varsayılan legend'i çizim
> alanının dışında (`x=1.02`) durur ve kendine **yatay margin rezerve eder**; uzun seri
> adlarıyla bu margin dar ekranda genişliğin ~%60'ını yiyordu. Asıl düzeltme: mobilde
> legend'i **yatay** ve grafiğin **üstüne** almak. İlk düzeltme (forecast uzatması)
> gerçek ama ikincil bir etkendi ve korunmuştur.

## Intent
Mobil görünümde grafik yatayda ekranın yalnızca sol ~yarısını kullanıyor; mumlar sola
sıkışıyor, sağ taraf büyük ölçüde boş (yalnızca ileriye uzanan AI tahmin çizgisi ve ızgara).
Neden, açılış x-ekseninin AI tahmini için ~15 bar geleceğe taşınmasıdır — dar mobil ekranda
bu boş gelecek alanı toplam genişliğin büyük bir oranını kaplar. Bu iyileştirmeyi telefondan
bakan son kullanıcı istiyor; çünkü sıkışmış mumlar fiyat hareketini okumayı zorlaştırıyor.
Başarı, mobilde açılışta mumların ekran genişliğinin büyük kısmını doldurması ve sağa doğru
yayılmasıdır. CLARIFY kararı: mobilde açılış penceresi tahminin **tamamını** kapsayacak kadar
geleceğe taşınmaz; yalnızca küçük bir ileri margin (mevcut `gap_multiplier`, ~5 bar) bırakılır.
Tahmin çizgisi **çizilmeye devam eder** ve sağa kaydırılarak tümüyle görülebilir — veri veya
özellik gizlenmez, yalnızca açılış x-aralığı değişir. Bilerek yapılmayan; masaüstü davranışını
değiştirmek, tahmin hesabına dokunmak veya y-eksenini değiştirmektir.

## Requirements
- Mobil modda grafik açılışında x-ekseni sağ kenarı, tahminin son tarihine kadar
  **uzatılmaz**; yalnızca son mumdan sonra küçük bir ileri margin bırakılır.
- Masaüstü modda mevcut davranış korunur: açılış x-aralığı tahminin son tarihini kapsar.
- Tahmin çizgisi (ve tüm diğer katmanlar) iki modda da figüre eklenmeye devam eder; mobilde
  yalnızca başlangıçta görünen pencere daralır, tahmin sağa kaydırınca erişilebilir kalır.
- Değişiklik yalnızca açılış x-aralığını (`xaxis.range`) etkiler; fiyat/tahmin değerlerini
  veya y-eksenini değiştirmez.

## Constraints
- Yalnızca `render_main_chart`'ın açılış x-aralığı hesabı değişir (`ui_components.py`).
- Mobil ileri margin = mevcut `gap_multiplier` (1wk: 3 bar, diğer: 5 bar) — yeni sabit
  gerekmez.
- **Yasak:** Masaüstü açılış aralığını değiştirmek; tahmin/analiz hesabını değiştirmek;
  y-ekseni aralığını değiştirmek; herhangi bir katmanı/veriyi gizlemek.
- **Kapsam dışı (V1):**
  - Y-ekseninin mobilde tahmin uç değerlerini kapsamaması (ayrı bir iyileştirme).
  - Masaüstü yerleşiminde herhangi bir değişiklik.
  - Otomatik genişlik/yön algısı, yeni kontroller.

## Context
- İlgili modül: `ui_components.py` → `render_main_chart` açılış x-aralığı
  (`zoom_end` hesabı, tahmin uzatması).
- İlgili spec: `specs/done/0001-...` (mobil görünüm modu) veya `specs/0001-...`.
- Referans: `stream.jpeg` (mumlar sola sıkışmış, sağ taraf gelecek boşluğu).

## Acceptance Criteria
> Her satır **tek başına test edilebilir** olmalı (bkz. `docs/testing.md`).
- [ ] **Mobil yatay doldurma:** show_pred açık ve tahmin son mumun ilerisine uzanırken,
      mobil modda üretilen figürün `xaxis.range[1]`'i tahminin son tarihinden (`f_dates[-1]`)
      **öncedir** (küçük ileri margin ile sınırlı).
- [ ] **Masaüstü regresyonu yok:** Aynı girdilerle masaüstü modda `xaxis.range[1]`
      tahminin son tarihini (`f_dates[-1]`) **kapsar** (mevcut davranış).
- [ ] **Tahmin gizlenmez:** Mobil modda tahmin serisi figüre yine eklenir (trace mevcut) ve
      tüm f_dates noktalarını içerir; yalnızca açılış x-aralığı daha dardır.
- [ ] **Değer/y-ekseni korunur:** Mod değişimi mumların ve tahminin değerlerini ve y-ekseni
      aralığını değiştirmez.
- [ ] **(rev 1) Mobil legend yatay ve grafiğin üstünde:** Mobil modda legend
      `orientation="h"`, `y >= 1.0`, `x = 0` ile konumlanır — çizim alanının sağında
      margin rezerve etmez, böylece mumlar tam genişliği kullanır.
- [ ] **(rev 1) Legend tema stili korunur:** Mobilde de legend arka planı/kenarlığı
      tema değerlerini taşır (sıfırdan yazılmaz).
- [ ] **(rev 1) Masaüstü legend regresyonu yok:** Masaüstünde legend varsayılan
      (dikey, sağ) konumda kalır.

## Definition of Done
- [ ] Tüm kabul kriterleri test/kanıtla karşılandı.
- [ ] Testler: `pytest` (`tests/test_chart_layout.py` genişletilir).
- [ ] Arayüz kanıtı: mobil ekran görüntüsü (mumlar genişliği dolduruyor) + masaüstü
      (değişmemiş).
- [ ] Yeşil pipeline (lint + test).
- [ ] Kullanıcıya teknik hata sızmıyor.
- [ ] PR şablonu dolduruldu, squash-merge (`docs/git.md`).

---

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 1 |
| Bulgu gerçek/gürültü oranı | 1/0 |
| Regresyon sayısı | 0 |
| Kaçan hata | 1 |

> Notlar: **Revizyon 1** — kök neden düzeltmesi (forecast uzatması değil, legend margini).
> **Düzeltme turu (1):** ilk uygulama cihazda etki etmedi, ikinci tur çözdü.
> **Bulgu (1/0):** cihaz testinden gelen tek gerçek bulgu ("değişmedi"), gürültü yok.
> **Kaçan hata (1):** İlk uygulama (mobilde forecast x-uzatmasının kaldırılması) yeşil
> testlerle `main`'e gitti ama kullanıcının gördüğü sorunu **çözmedi** — kök neden yanlış
> teşhis edilmişti. Ders: ekran görüntüsü "sağ taraf boş" derken, boşluğun *çizim alanı
> içinde* mi (eksen aralığı) yoksa *çizim alanı dışında* mı (margin/legend) olduğu
> ölçülmeden teşhis konmamalı; Python testleri Plotly'nin tarayıcıda hesaplanan
> margin/legend yerleşimini yakalayamaz, bu yüzden cihaz teyidi şart.
>
> **Kapanış:** Cihazda doğrulandı (mumlar tam genişliği kullanıyor) — DoD'un görsel kanıt
> maddesi karşılandı, spec `done/` altına alındı.
