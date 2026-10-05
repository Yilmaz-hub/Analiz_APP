# Spec: 0011 — Pozisyon Sembolü ve Varlık Adı Önerisi

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer. Revizyon 1 (2026-10-05).
> Durum: **TASLAK.** Kaynak: Takım Yöneticisi (2026-10-05): "Sonradan eklediklerim varlık listesinden silinmiş, ondan hesaplanamıyormuş. Varlık adından değil varlık kodundan kontrol etse daha iyi. Varlık eklerken kodu yazınca adını öneri olarak çıkarsa iyi olur; listede `Bitcoin (BTC)` yazıyor, benim eklediğim `link` yazıyor."

## Intent
Pozisyonun fiyatı, varlık listesinden bağımsız olarak pozisyonun kendi sembolüyle alınsın. Yeni varlık eklerken kod yazılınca tutarlı bir ad önerilsin.

## Requirements
- **R01:** Yeni pozisyon kaydı, açıldığı andaki varlığın sembolünü `Sembol` alanında taşır.
- **R02:** Fiyat yolu önce kaydın `Sembol` alanını kullanır; yoksa (eski kayıt) ad varlık listesine eşlenir (spec 0010).
- **R03:** Varlık ekleme formunda Yahoo kodu yazılınca ad önerilir: önce listede/varsayılanda aynı sembolün adı, sonra bilinen kripto adı (`Chainlink (LINK)`). Bilinmeyen kod için ad uydurulmaz. `LINKUSD` ve `LINK-USD` aynı öneriyi verir.
- **R04:** Öneri yalnız öneridir; "Öneriyi kullan" ad alanını doldurur, kullanıcı değiştirebilir. Kayıtlı eski pozisyonlar yeniden yazılmaz.

## Acceptance Criteria
- [ ] **AC05 — Ad önerisi, R03:** `LINKUSD`/`link-usd` → `Chainlink (LINK)`; `BTC-USD` → varsayılandaki `Bitcoin (BTC)`; bilinmeyen kod ve boş kod → öneri yok; ad zaten listedeyse öneri yok.
- [ ] **AC06 — Sembol kaydı, R01:** Yeni pozisyon `Sembol` alanını taşır.
- [ ] **AC07 — Silinmiş varlık, R02:** Varlık listeden silinmiş olsa da `Sembol` taşıyan pozisyon fiyatlanır.
- [ ] **AC08 — Ekranda öneri, R03/R04:** Kod yazılınca "Önerilen ad" görünür; "Öneriyi kullan" ad alanını doldurur.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü
- [ ] Yayında: varlık eklerken `LINKUSD` yazınca `Chainlink (LINK)` önerilir

## Bilinen sınır
Ad listesi kapalı (~40 yaygın kripto); çevrim içi ad sorgusu yok. Hisse ve döviz için yalnız varsayılan listedeki adlar önerilir. Eski pozisyonlarda `Sembol` yok; onlar ad eşlemesine düşer. Ekleme formu artık `st.form` değil (canlı öneri için); etiketler ve "Listeye Ekle" düğmesi aynıdır.

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | Silinmiş varlığın pozisyonu fiyatsız kalıyordu; silme koruması her yolda geçerli olmayabilir (neden doğrulanamadı) |
