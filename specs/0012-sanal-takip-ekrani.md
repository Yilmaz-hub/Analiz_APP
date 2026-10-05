# Spec: 0012 — Sanal Takip Ekranının Sadeleştirilmesi

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-05).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi (2026-10-05): "İleri dönem sanal takip ekranında yazanlar aşırı kalabalık, neyin ne olduğu anlaşılmıyor. Anlaşılır tasarla."

## Sorun (ekran görüntüsüyle doğrulandı)
- Aynı "Son başarılı çalışma / Son çalışma notu" iki kez ve ham ISO zamanıyla görünüyor; not, 12 varlığın `miktar adımı/varsayımlar eksik` cümlesini tek satırda tekrarlıyor.
- Her varlık için 5–7 düz `caption` satırı ("yetersiz kanıt (izlenen gün 0/90, …)", ISO zamanlı kararlar, "bilinmiyor" dolu varsayım listesi) alt alta; neyin ne olduğu ve ne yapılması gerektiği yok.
- Özelliğin ne yaptığı hiçbir yerde tek cümleyle yazılı değil.

## Intent
Ekran üç soruyu cevaplasın: **Çalışıyor mu? Ne kaydediyor? Benden bir şey bekleniyor mu?** Ayrıntı isteğe bağlı olsun.

## Requirements
- **R01 — Durum:** Üstte tek satır durum: çalışıyor (yeşil) / durmuş olabilir (uyarı) / henüz çalışmadı (bilgi). Zaman İstanbul saatiyle ve "3 saat önce" biçiminde yazılır; ham ISO gösterilmez. Başarısız son çalışma ayrı uyarıdır.
- **R02 — Açıklama:** Özelliğin ne yaptığı tek cümlede yazılır (her gün mum kapanınca AL/BEKLE/SAT kaydedilir; yeterli gün ve işlem birikince sanal sonuç değerlendirilir).
- **R03 — Not gruplama:** Koşucu notundaki "miktar adımı/varsayımlar eksik" satırları tek özet cümlede toplanır (kaç varlık, ne anlama geliyor, ne yapmalı: işlem varsayımlarını girmek); ayrıntı katlanır bölümdedir. Veri alınamadı gibi diğer notlar kaybolmaz.
- **R04 — Tablo:** Ayrıntı açıldığında varlık başına **tek satırlık** tablo: son karar (AL/BEKLE/SAT + gün), izlenen gün, kapanmış işlem, piyasa koşulu (3 üzerinden), durum (Yeterli kanıt / Birikiyor).
- **R05 — Seçili varlık:** Tablonun altında seçilen varlığın son 5 kararı (gün, karar, zamanında/sonradan, kaynak), eksik görünen gün ve revizyon uyarıları, kayıtlı varsayımlar (`bilinmiyor` → `girilmedi`) katlanır bölümde.
- **R06:** Hesap ve kayıt mantığı değişmez; yalnız gösterim. `build_forward_view` (düz satır üretici) eski testler için korunur.

## Acceptance Criteria
- [ ] **AC01 — Zaman biçimi, R01:** `friendly_time` İstanbul saatiyle "5 Eki 08:52 (3 saat önce)" üretir.
- [ ] **AC02 — Durum satırı, R01:** Hiç çalışmadıysa bilgi; 2 günden eskiyse uyarı; sağlıklıysa "Takip çalışıyor"; başarısız son çalışma ayrı uyarı verir.
- [ ] **AC03 — Not gruplama, R03:** 11 varlığın eksik-varsayım notu tek özet cümle ve tek ayrıntı satırı olur; `veri alınamadı` gibi diğer notlar ayrıntıda korunur.
- [ ] **AC04 — Özet tablo, R04:** Her varlık-sürüm çifti için bir satır; sayaçlar doğru; en çok `MAX_PAIRS` satır.
- [ ] **AC05 — Varlık ayrıntısı, R05:** Son 5 karar okunur biçimde; `bilinmiyor` değeri `girilmedi` yazılır.
- [ ] **AC06 — Ekran, R01–R05:** Ekranda ham ISO zamanı ve tekrar eden uzun not görünmez; açıklama cümlesi, özet cümle ve tablo görünür.

## Bilinen sınır / karar bekleyen
- Panel **karar kaydı ve sanal işlem sayacı** gösterir; "tahmin doğru çıktı mı" (isabet oranı) hesaplanıp gösterilmez. Bu yeni bir ölçüttür (hangi ufukta, hangi kurala göre doğru sayılır) ve ayrı spec + karar ister.
- `EURUSD=X` gibi tanınmayan piyasalar koşucuda atlanır (spec 0005); bu spec kapsamında değil.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü; yayında kullanıcı kontrolü

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | Ekran yalnız `build_*` metin satırlarıyla sınanmıştı; okunabilirlik hiç sınanmadı |
