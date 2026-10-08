# Spec: 0017 — Dürüst Teşhis: Medyan, 1R Eşiği, Kazanan/Kaybeden Ayrımı, Giriş Koşulu

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-08).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi (2026-10-08) ETH sonucu: teşhis "ortalama %15,5 yükseğe çıktı, 4/6 kâra geçip zararla kapandı" dedi; ama kâr koruma karşılaştırmasında üç kuralın üçü de V1'den kötü çıktı (2R −144 $, kademeli −143 $, iz süren −281 $). Teşhis yanıltıcıydı; kullanıcı düzeltmeye "Yap" dedi.

## Kök neden
- Ortalama, birkaç büyük kazanan işlemle şişiyor; tipik işlemi göstermiyor.
- "Kâra geçti" girişin bir kuruş üstünü bile sayıyor; anlamlı kâr değil.
- Kazanan ve kaybedenler ayrılmadığı için zayıflığın (kaybeden girişler mi, geri verilen kâr mı) hangisi olduğu görünmüyor.

## Requirements
- **R01:** Teşhis değerleri ortalama değil **medyan**dır.
- **R02:** "Kâra geçip zararla kapandı" yalnız en az **1R** (giriş − V1 başlangıç stopu; stop karar günü ATR'siyle) kâr görülen işlemleri sayar. R hesaplanamıyorsa sayılmaz ve "hesaplanamıyor" yazılır.
- **R03:** Kazanan ve kaybedenler ayrı satırda: adet, medyan kapanış, medyan en iyi seviye, giriş koşulu dağılımı (yükselen / düşen / yatay). Giriş koşulu yalnız girişe karar verilen güne kadarki veriyle hesaplanır (sızıntı yok).
- **R04:** İşlem işlem tablo (giriş, çıkış, sonuç, neden, kapanış %, en iyi %, en iyi R, giriş koşulu).
- **R05:** 30'dan az kapanmış işlemde "desen için az" uyarısı. Strateji kuralı ve mevcut hesaplar değişmez.

## Acceptance Criteria
- [ ] **AC01 — Medyan, R01:** Bir büyük kazanç ortalamayı şişirse de medyan tipik işlemi gösterir.
- [ ] **AC02 — 1R eşiği, R02:** Girişin hemen üstünü gören işlem sayılmaz; 1R üstünü gören sayılır.
- [ ] **AC03 — Ayrım ve koşul, R03/R05:** Kazanan/kaybeden satırları ve giriş koşulu; koşul yalnız karar günü verisiyle; az işlem uyarısı.
- [ ] **AC04 — Tablo, R04:** Her kapanmış işlem için bir satır.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | 0016 teşhisi ortalama ve sıfır eşiği kullandı; kullanıcıyı yanlış nedene (kâr alma) yöneltti |
