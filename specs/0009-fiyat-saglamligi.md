# Spec: 0009 — Eski Kayıtların Fiyatlanması ve Portföy Önceliği

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist. Revizyon 1 (2026-10-05).
> Durum: **TASLAK — acil düzeltme.** Kaynak: Takım Yöneticisi'nin yayındaki bildirimi (2026-10-05): portföy tablosunda `link`, `Hbar`, `ONS ALTIN`, `EUR/USD` kâr/zarar "hesaplanamıyor"; "Fiyat kaynağı ayrıntısı" bu varlıklar için **"kaynaklara henüz sorulmadı"** diyor; "sanal takibi açınca yine portföy yüklenmedi".

## Kök neden (kodla doğrulandı)
- **0007 regresyonu:** `app.run_position_prices` yalnız `Status` alanı tam `"ACTIVE"`/`"PENDING"` olan kayıtların fiyatını istiyordu. Uygulamanın tek sınıflandırıcısı (`positions.classify_position`) ise `Status` alanı olmayan eski kayıtları ve kalan miktarı olan "kapanmış" kayıtları da **aktif** sayar. Bu kayıtlar tabloya giriyor ama fiyat kaynağına hiç sorulmuyordu. Yeni akışla açılan pozisyonlar (`ETH`, `SOL`) `Status` taşıdığı için çalışıyordu.
- **Sıra:** İleri takip ve kağıt ticaret panelleri portföy bölümünden **önce** çiziliyordu; ağır panel açıkken (çok sorgu) portföy bekliyordu.

## Intent
Tabloda gösterilen her pozisyonun fiyatı gerçekten sorulsun; portföy, ağır panellerden önce gelsin.

## Requirements
- **R01:** Fiyat sorulacak pozisyon kümesi, tablolara girenlerle **aynı** sınıflandırıcıdan (aktif + bekleyen) gelir; ayrı bir `Status` karşılaştırması yoktur.
- **R02:** Fiyatı alınamayan pozisyon için ayrıntı, varlık adı varlık listesinde yoksa bunu söyler (adı değişmiş ya da silinmiş olabilir).
- **R03:** Portföy ve cüzdan bölümü, ileri takip ve kağıt ticaret panellerinden önce çizilir.
- **R04:** İleri takip paneli ayrıntıyı en çok 6 varlık-sürüm çifti için sorgular; kalanı için sayıyı belirtir.

## Acceptance Criteria
- [ ] **AC09 — Eski kayıtlar fiyatlanır, R01:** `Status` alanı olmayan ya da kalan miktarı olan kaydın fiyatı sorulur ve tabloda "canlı" görünür; tabloya giren hiçbir satır fiyat sorulmadan kalmaz.
- [ ] **AC10 — Portföy önce, R03:** Ana betikte portföy bölümü, ileri takip ve kağıt ticaret panellerinden önce gelir.
- [ ] **AC11 — Çift sınırı, R04:** İleri takip paneli ayrıntı sorgusunu en çok 6 çift için yapar; fazlası için "N çift daha" notu gösterilir.
- [ ] **AC12 — Eksik varlık notu, R02:** Fiyatı alınamayan satırın adı varlık listesinde yoksa kaynak ayrıntısı bunu açıkça söyler.

## Definition of Done
- [ ] Testler yeşil (tam suite + `-m perf` + `-m browser`)
- [ ] Ayrı QA oturumu kabulü
- [ ] Yayında doğrulama: `link`/`Hbar` fiyatı gelir; sanal takip açılınca portföy beklemez

## Bilinen sınır
- `ONS ALTIN` ve `EUR/USD` yalnız Yahoo'dan fiyat alır; Yahoo bu sunucudan yanıt vermiyorsa bu düzeltme onları fiyatlamaz. Kaynak ayrıntısı (artık gerçekten sorulduktan sonra) bunu gösterecek; ikinci kaynak ayrı karardır.

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 1 — 0007'nin `Status` filtresi (bu spec düzeltir) |
| Kaçan hata | 0007 testleri yalnız yeni akışla açılmış (`Status` taşıyan) pozisyonları kullandı; eski kayıt biçimi sınanmadı |
