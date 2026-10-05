# Spec: 0008 — Yayında Sayfa Yükleme Hızı (kayıt deposu gidiş-dönüşleri ve uyuyan bağlantı)

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist. Revizyon 1 (2026-10-05).
> Durum: **TASLAK — acil düzeltme.** Kaynak: Takım Yöneticisi'nin yayındaki bildirimi (2026-10-05, ekran görüntüsü): "Reset sonrası portföy yüklenmiyor / yüklenmesi uzun sürüyor." Görüntüde sayfa "İleri Dönem Sanal Takip" bölümünde kesiliyor ve sağ üstte "Stop" (betik çalışıyor) görünüyor; portföy bölümü bu noktadan sonra geliyor. Yayın günlüklerine erişilemedi; kök neden ölçümle sınırlandı (aşağıda).

## Intent
Yeniden başlatma ya da uzun hareketsizlik sonrası portföy bölümü uzak veritabanı (Neon) gecikmesi yüzünden bekletilmesin; ekran açıkça gerekmeyen kayıt sorgularını her tıklamada yeniden yapmasın.

## Ölçüm (taklit gecikme 150 ms/sorgu, 5 varlık, 4 pozisyon)
- Sıcak çalıştırma 3,6 sn; bunun **2,7 sn'si** ileri takip panelinde. Panelin 2,3 sn'si her çağrıda yeniden çalışan 12 `CREATE TABLE IF NOT EXISTS` sorgusu. Gidiş-dönüş süresi arttıkça (Neon uyanma, bölge farkı) bu doğrusal büyür.
- Ekran kapalı bölüm (expander) içeriği de her çalıştırmada hesaplanıyor: görünmeyen içerik portföyden önce çalışıyor.

## Requirements
- **R01:** İleri takip tabloları süreç başına bir kez oluşturulur; sonraki çağrılar ek DDL sorgusu yapmaz.
- **R02:** İleri takip durumu, kullanıcı "göster" demeden hesaplanmaz; varsayılan açılışta ileri takip tablolarına sorgu gitmez.
- **R03:** Veritabanı bağlantı havuzu uyuyan/kopmuş bağlantıyı kullanmadan önce sınar ve eskiyen bağlantıyı yeniler (Neon boşta bağlantıyı kapatır).

## Constraints
- Kayıt hiçbir koşulda silinmez ya da yeniden oluşturulmaz; tablo şeması değişmez.
- Teknik hata metni ve parola kullanıcıya sızmaz ([conventions.md](../docs/conventions.md)).

## Acceptance Criteria
- [ ] **AC01 — Tablolar bir kez, R01:** İleri takip işlevleri art arda çağrıldığında ilk çağrıdan sonra `CREATE TABLE` sorgusu çalışmaz.
- [ ] **AC02 — Tembel panel, R02:** Ana ekran varsayılan açılışta `forward_*` tablolarına sorgu yapmaz; "Sanal takip durumunu göster" açılınca panel içeriği görünür.
- [ ] **AC03 — Bağlantı sınaması, R03:** Havuz bağlantıyı kullanmadan önce sınar (`pool_pre_ping`) ve en çok 5 dakikada bir yeniler (`pool_recycle`).

## Definition of Done
- [ ] Testler yeşil (tam suite + `-m perf` + `-m browser`)
- [ ] Ayrı QA oturumu kabulü; QA'dan önce merge yok
- [ ] Yayında doğrulama: yeniden başlatma sonrası portföy bölümü birkaç saniyede gelir

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | Ölçülmedi |
| Kaçan hata | Yayındaki gerçek gecikme ölçülemedi; taklit gecikmeyle sınırlandı |
