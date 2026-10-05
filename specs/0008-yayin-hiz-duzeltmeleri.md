# Spec: 0008 — Yayında Sayfa Yükleme Hızı (kayıt deposu gidiş-dönüşleri ve uyuyan bağlantı)

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist. Revizyon 1 (2026-10-05).
> Durum: **TASLAK — acil düzeltme.** Kaynak: Takım Yöneticisi'nin yayındaki bildirimi (2026-10-05, ekran görüntüsü): "Reset sonrası portföy yüklenmiyor / yüklenmesi uzun sürüyor." Görüntüde sayfa "İleri Dönem Sanal Takip" bölümünde kesiliyor ve sağ üstte "Stop" (betik çalışıyor) görünüyor; portföy bölümü bu noktadan sonra geliyor. Yayın günlüklerine erişilemedi; kök neden ölçümle sınırlandı (aşağıda).

## Intent
Yeniden başlatma ya da uzun hareketsizlik sonrası portföy bölümü uzak veritabanı (Neon) gecikmesi yüzünden bekletilmesin; ekran açıkça gerekmeyen kayıt sorgularını her tıklamada yeniden yapmasın.

## Ölçüm (taklit gecikme 150 ms/sorgu, 5 varlık, 4 pozisyon)
- Sıcak çalıştırma 3,6 sn; bunun **2,7 sn'si** ileri takip panelinde. Panelin 2,3 sn'si, her `forward_tracker` çağrısında yeniden gönderilen 4 `CREATE TABLE IF NOT EXISTS` ifadesiydi (bir ekran çalıştırmasında 3–4 çağrı ⇒ 12–15 sorgu). Gidiş-dönüş süresi arttıkça (Neon uyanma, bölge farkı) bu doğrusal büyür.
- Ekran kapalı bölüm (expander) içeriği de her çalıştırmada hesaplanıyor: görünmeyen içerik portföyden önce çalışıyor.

## Requirements
- **R01:** İleri takip tabloları süreç başına bir kez oluşturulur; sonraki çağrılar ek DDL sorgusu yapmaz.
- **R02:** İleri takip durumu, kullanıcı "göster" demeden hesaplanmaz; varsayılan açılışta ileri takip tablolarına sorgu gitmez.
- **R04:** Koşucunun sağlığı (son çalışma başarısız mı, kaç gündür çalışmadı) panel kapalıyken de iki ucuz sorguyla görünür; kesinti ekranda gizli kalmaz (0005 AC38/AC99).
- **R03:** Veritabanı bağlantı havuzu uyuyan/kopmuş bağlantıyı kullanmadan önce sınar ve eskiyen bağlantıyı yeniler (Neon boşta bağlantıyı kapatır).

## Constraints
- Kayıt hiçbir koşulda silinmez ya da yeniden oluşturulmaz; tablo şeması değişmez.
- Teknik hata metni ve parola kullanıcıya sızmaz ([conventions.md](../docs/conventions.md)).

## Acceptance Criteria
- [ ] **AC01 — Tablolar bir kez, R01:** İleri takip işlevleri art arda çağrıldığında ilk çağrıdan sonra `CREATE TABLE` sorgusu çalışmaz.
- [ ] **AC02 — Tembel panel, R02:** Ana ekran varsayılan açılışta `forward_*` tablolarına sorgu yapmaz; "Sanal takip durumunu göster" açılınca panel içeriği görünür.
- [ ] **AC04 — Kesinti görünürlüğü, R04:** Başarısız son çalışma ve 2 günden uzun süredir çalışmayan koşucu uyarısı, toggle açılmadan, en çok 2 sorguyla görünür; sağlıklı koşucuda uyarı yoktur ve son başarılı çalışma görünür.
- [ ] **AC03 — Bağlantı sınaması, R03:** Havuz bağlantıyı kullanmadan önce sınar (`pool_pre_ping`) ve en çok 5 dakikada bir yeniler (`pool_recycle`).

## Bilinen sınırlar ve kabul edilen takaslar (QA turu 1)
- **Bayrak süreç ömrü boyunca geçerlidir:** tablo dışarıdan silinirse süreç yeniden başlayana kadar "İleri takip kayıtlarına erişilemedi" görünür (şema değişmediği sürece beklenmez).
- **Eşzamanlı ilk çağrı:** iki iş parçacığı birden `CREATE TABLE IF NOT EXISTS` gönderebilir; hata olursa bayrak set edilmez ve sonraki çağrı yeniden dener (önceki davranıştan daha az riskli).
- **`pool_pre_ping` maliyeti:** Postgres'te havuzdan her alışta bir gidiş-dönüş ekler (150 ms'de ≈0,3 sn sıcak, ≈1,3 sn soğuk); Neon'un kapattığı bağlantıyı yakalamak için kabul edilen takas.
- **Testin özel öznitelik bağımlılığı:** AC03 testi `pool._pre_ping` ve `pool._recycle` okur (SQLAlchemy 2.0.52); sürüm yükseltmede gözden geçirilir.

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
