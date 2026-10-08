# Spec: 0016 — Geçmiş Test Teşhisi: Giriş, Kâr Alma, Stop, Girmeme

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 2 (2026-10-08; QA: ısınma ve çıkış günü).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi (2026-10-08): "İyi strateji ne zaman alınacağı ve satılacağı, kâr alınacak yer, stop olacağı yeri bilmek, ayrıca ne zaman almayacağını bilmektir." Önerilen "önce ölç" yoluna onay; BTC'de işlem tutarı 10000 ile getiri −%9.

## Intent
Stratejiyi değiştirmeden önce zayıflığın nerede olduğunu (kötü giriş mi, kâr almamak mı, stop mu, dışarıda kalma mı) geçmiş test sonucundan ölçmek.

## Requirements
- **R01 — Çıkışlar:** Kapanmış işlemlerin çıkış nedeni sayılır (stop / SAT sinyali / diğer).
- **R02 — Kâr alma:** Her kapanmış işlem için işlem süresince görülen en yüksek seviye (çıkış günü yalnız çıkış fiyatıyla; o günün sonradan oluşan yükseği sayılmaz) ve kapanış, girişe göre yüzde; ortalamaları ve aradaki fark ("geri verilen"); kâra geçip zararla kapanan işlem sayısı.
- **R03 — Girmeme:** Günlük kapanışlarla, pozisyondayken ve dışarıdayken varlığın bileşik getirisi; dışarıda düşüş varsa "korudu", yükseliş varsa "kaçırdı" yorumu. Açık pozisyon günleri pozisyonda sayılır; ilk karardan önceki ısınma günleri sayılmaz.
- **R04:** Strateji kuralı ve mevcut hesaplar değişmez; Decimal; geçersiz/kısa veride None, NaN istisna üretmez; yaklaşık ölçüm ve maliyet hariç olduğu yazılır.

## Acceptance Criteria
- [ ] **AC01 — Çıkış nedenleri, R01**
- [ ] **AC02 — Geri verilen kâr, R02:** %20 tepe / %-5 kapanış → fark %25, kâra geçip zararla kapanan 1.
- [ ] **AC03 — Dışarıda kalma, R03**
- [ ] **AC04 — Açık pozisyon ve geçersiz veri, R03/R04**
- [ ] **AC05 — Ekran:** Günlük geçmiş test sonrası teşhis başlığı ve metrikleri görünür.

## Kapsam dışı
Kâr alma / iz süren stop / girmeme filtresi eklemek (strateji değişikliği). Teşhis sonucuna göre ayrı spec ile aday strateji olarak denenir.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | Geçmiş test yalnız toplam getiriyi gösteriyordu; zayıflığın kaynağı ölçülmüyordu |
