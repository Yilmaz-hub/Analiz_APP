# Spec: 0007 — Yayın Ortamı Düzeltmeleri (kayıt bağlantısı ve fiyat bekleme süresi)

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist. Revizyon 1 (2026-10-04).
> Durum: **TASLAK — acil düzeltme.** Kaynak: Takım Yöneticisi'nin yayındaki bildirimi (2026-10-04): "kâr al kısmında yüzde yazdım, fiyatlar bozuldu; yeniden başlattım, portföy gelmiyor; backtest ve ileri sanal takip 2–3 kez yazılmış". Yayın günlüklerine erişilemediği için kök neden kesin değildir; bu spec, **gözlemlenen belirtileri açıklayan iki doğrulanabilir zayıflığı** kapatır. Kesin neden, düzeltmeden sonra yayında doğrulanır (AP-05).

## Intent
Uygulama yeniden başlatıldığında kayıtlar (portföy, varlıklar) bağlantı adresinin yazılış biçimi yüzünden sessizce kaybolmasın; fiyat kaynakları yavaş ya da erişilemezken sayfa dakikalarca donmasın, portföy ve paneller bekletilmeden gelsin.

## Requirements
- **R01:** `db_url` Neon/Postgres panelinden kopyalandığı biçimiyle (`postgres://…`, `postgresql://…`, başında/sonunda boşluk ya da tırnak) çalışır; uygulama sürücüsü (`psycopg`) kendiliğinden seçilir. Sürücüsü açıkça yazılmış adres ve SQLite adresi değişmez.
- **R02:** Bağlantı kurulamazsa kullanıcıya kayıtların silinmediği ve adresin kontrol edilmesi gerektiği söylenir; teknik metin ve parola sızmaz.
- **R03:** Bir çalıştırmada tüm pozisyonların fiyatları **eşzamanlı** ve **toplam süre bütçesi içinde** alınır; bütçe aşılırsa kalan fiyatlar "fiyat alınamadı" olur (R12, spec 0006), sayfa beklemez. Geç dönen fiyat sonraki çalıştırmada önbellekten gelir.
- **R04:** Duruk (stop) teması uyarısı da aynı fiyat kümesini kullanır; ayrı, sıralı ağ çağrısı yapmaz.

## Constraints
- Para ve yüzde `Decimal` ([conventions.md](../docs/conventions.md)); bu spec fiyat değerlerinin türünü değiştirmez.
- Kayıt hiçbir koşulda silinmez ya da yeniden oluşturulmaz.

## Acceptance Criteria
- [ ] **AC01 — Adres biçimi, R01:** `postgres://…` ve `postgresql://…` adresleri `postgresql+psycopg://…` olarak kullanılır; `postgresql+psycopg://…`, `postgresql+psycopg2://…` ve `sqlite:///…` değişmez; çevreleyen boşluk ve tırnak atılır.
- [ ] **AC02 — Bağlantı hatası, R02:** Uzak veritabanına bağlanılamadığında kenar çubuğu "kayıtlarınız silinmedi, bağlantı adresini kontrol edin" der; parola ve teknik metin görünmez; uygulama çökmez.
- [ ] **AC03 — Eşzamanlı fiyat, R03:** 6 pozisyonun her biri 1 sn süren fiyat kaynağıyla toplam bekleme ≈ 1 sn'dir (6 sn değil).
- [ ] **AC04 — Süre bütçesi, R03:** Bütçeden uzun süren fiyat kaynağı sayfayı bütçeden fazla bekletmez; o pozisyon "fiyat alınamadı" gösterir, diğerleri fiyatlarıyla görünür.
- [ ] **AC05 — Ekranda donma yok, R03:** Fiyat kaynağı 5 sn/pozisyon yavaşken ana ekran bütçe sınırı + payı içinde tamamlanır ve portföy tablosu gelir.
- [ ] **AC06 — Tek fiyat kümesi, R04:** Stop teması uyarısı verilen fiyat kümesiyle üretilir; fiyat kaynağı ek kez çağrılmaz.

## Definition of Done
- [ ] Testler yeşil (tam suite + `-m perf`)
- [ ] Ayrı QA oturumu kabulü; QA'dan önce merge yok
- [ ] Yayında doğrulama: yeniden başlatma sonrası portföy gelir; kenar çubuğu rozeti "kalıcı" der

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | Ölçülmedi |
| Kaçan hata | Yayında kök neden doğrulanamadı; yalnız belirtileri açıklayan zayıflıklar kapatıldı |
