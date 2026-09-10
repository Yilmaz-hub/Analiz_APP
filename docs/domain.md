# Alan (domain) — Ortak Dil ve İş Kuralları

> **Ubiquitous language**: kodda, spec'te ve konuşmada aynı terim kullanılır.
> Terim değişirse burada değişir; başka yerde tekrar tanımlanmaz.

## Sözlük

| Terim (TR) | Kod adı (EN) | Tanım |
|------------|--------------|-------|
| Ürün | `Product` | Satılabilir kalem; fiyatı `decimal`. |
| Sipariş | `Order` | Bir müşterinin verdiği, kalemlerden oluşan alım. |
| Sipariş Kalemi | `OrderLine` | Bir siparişteki tek ürün satırı (adet + birim fiyat). |
| Kupon | `Coupon` | Sipariş tutarını düşüren, kurallı indirim aracı. |
| İndirim | `Discount` | Kupon ya da kampanyadan doğan tutar/yüzde düşüşü. |
| İade | `Return` | Teslim sonrası ürün/sipariş geri dönüş talebi. |
| Tutar | `Money` (decimal) | Para değeri; **her zaman `decimal`**, para birimiyle. |

## İş Kuralları

### Sipariş
- Bir sipariş en az bir kalem içermelidir; boş sipariş oluşturulamaz.
- Sipariş toplamı = kalem (birim fiyat × adet) toplamı − uygulanan indirimler; sonuç `decimal`.
- Durum akışı: `Created → Paid → Shipped → Delivered`; ayrıca `Cancelled`.
- `Delivered` olmadan `Return` başlatılamaz.

### Kupon
- Kuponun geçerlilik tarihi ve (varsa) minimum sipariş tutarı vardır.
- Kupon tipi ya **yüzde** (`decimal`, ör. `%10`) ya da **sabit tutar** (`decimal`).
- Süresi geçmiş, koşulu tutmayan ya da daha önce kullanılmış kupon reddedilir.
- Bir siparişe en fazla bir kupon uygulanır (aksi ayrı spec konusudur).
- İndirim, sipariş toplamını sıfırın altına düşüremez.

### İade
- İade yalnızca `Delivered` siparişler için ve iade penceresi içinde açılabilir.
- İade tutarı, iade edilen kalemlerin ödenmiş tutarını aşamaz.
- İade onaylanınca ilgili tutar geri ödenir; sipariş durumu iz kaydında tutulur.

> Yeni bir iş kuralı ekleniyorsa: önce bir mini-spec'te kabul kriteri olarak yazılır,
> sonra buraya ortak dil olarak işlenir.
