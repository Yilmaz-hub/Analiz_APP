# Test Yaklaşımı

## Backend

- Çatı: **xUnit**.
- Kural: **her kabul kriteri bir test.** Spec'teki her Acceptance Criteria satırı en az bir
  testle karşılanır.
- Adlandırma deseni: **`Metot_Durum_BeklenenSonuc`**
  - ör. `ApplyCoupon_WhenExpired_ReturnsRejected`
  - ör. `CreateOrder_WithNoLines_Fails`
- Para/yüzde iddiaları `decimal` ile yapılır (kayan nokta karşılaştırması yok).
- Testler bağımsız ve tekrarlanabilir; dış servis mock/fake ile izole edilir.
- Test projesi kapsanan modüle göre ayrılır (ör. `Ordering.Tests`).

## Arayüz (Frontend)

- **Her mini-spec kriterine görsel kanıt**: ilgili ekranın **ekran görüntüsü** PR'a eklenir.
- Yükleme / boş / hata durumları da kanıtlanır (bkz. `frontend.md`).
- **Kritik akışa smoke test**: en az bir uçtan uca "mutlu yol" otomasyonu (ör. ürün listele →
  sepete ekle → sipariş oluştur).
- **CI'da lint + build** zorunlu (ayrıntı: `frontend.md`).

## Genel

- Yeşil pipeline olmadan PR merge edilmez (bkz. `git.md`).
- Bir hata bulunduğunda önce hatayı gösteren test yazılır, sonra düzeltme yapılır (kaçan
  hata SCORECARD'a işlenir — `specs/TEMPLATE.md`).
