# Spec: 0015 — Geçmiş Test Sonucunda Al-Tut Karşılaştırması

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 2 (2026-10-08; QA: ısınma dönemi).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi (2026-10-08): al-tut karşılaştırması önerisine "olur"; "başka birkaç varlıkta da denedim, onlarda da %1 gibi düşük bir getiri verdi."

## Sorun
"Toplam Getiri" sermayeye göredir; işlem tutarı 1000, sermaye 10000 iken strateji her alımda sermayenin yalnız %10'unu kullanır, bu yüzden iyi ya da kötü her sonuç ~%1 ölçeğinde görünür. Ekranda stratejinin piyasaya göre iyi mi kötü mü olduğunu gösteren bir referans yok. Mevcut al-tut karşılaştırması "Güvenilir Performans Raporu"nun içinde, ayrı değerlendirme diliminde ve tüm sermayeyle (strateji ile eşit olmayan maruziyetle) yapılıyor.

## Requirements
- **R01:** Günlük geçmiş test sonucunun hemen altında üç değer: strateji (sermayeye göre), aynı işlem tutarıyla al-tut (sermayeye göre), varlığın fiyat değişimi (tüm sermayeyle al-tut).
- **R02:** Al-tut, stratejinin ilk alım yapabileceği anda alır: ilk kararın (ısınma dönemi sonrası) bir sonraki mumunun açılışı; başlangıç tarihi ekranda yazılır. Son kapanışta değerlenir; maliyet hariçtir ve yaklaşıktır (strateji miktarı adıma yuvarlar). İşlem tutarı sermayeden büyükse karşılaştırma gösterilmez.
- **R03:** Tek cümlelik hüküm (daha iyi / daha kötü / aynı) ve işlem tutarı / sermaye oranı açıklaması.
- **R04:** Geçersiz girdide (kısa veri, sıfır/NaN fiyat, sıfır tutar, eksik sonuç) hesap yapılmaz, istisna fırlamaz. Hesap Decimal'dir. Mevcut hesaplar değişmez.

## Acceptance Criteria
- [ ] **AC01 — Ölçek ve başlangıç, R01/R02:** Fiyat %50 artışta, %10 maruziyetle al-tut sermayeye göre %5; ısınmalı veride al-tut ilk karardan sonraki açılışta başlar.
- [ ] **AC02 — Düşen piyasa, R03:** Piyasa düşerken stratejinin daha az kaybı "daha iyi" yazılır.
- [ ] **AC03 — Geçersiz girdi, R04:** None döner, istisna yok.
- [ ] **AC04 — Ekran, R01/R03:** Geçmiş test sonrası üç metrik ve oran açıklaması görünür.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | Sonuç ekranı getiriyi referanssız ve maruziyet açıklamasız gösteriyordu |
