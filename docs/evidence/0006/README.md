# Spec 0006 — ekran kanıtı (QA bulgusu K6)

Görüntüler gerçek `app.py` ile üretildi: ağ kaynakları taklit edilmiş, kayıt deposu geçici klasöre
yönlendirilmiş (`harness.py`); gerçek kayıtlara dokunulmaz. Yeniden üretim:
`python docs/evidence/0006/capture.py` (Playwright ve önceden kurulu Chromium gerekir).

| Dosya | Kanıtlanan |
|---|---|
| `kismi-satis-yuzde.png` | AC17 — %50 düğmesi yüzdeyi ve satılacak miktarı doldurur; ekran kısmi satışı önermez (AC18) |
| `kismi-satis-sonrasi.png` | AC17, AC06 — onaydan sonra pozisyon kalan 5 adetle aktif listede kalır |
| `fiyat-guvenilirligi.png` | AC27, AC30 — `LINKUSD` canlı fiyatla kâr/zarar gösterir; fiyatı alınamayan `HBARUSD` "hesaplanamıyor / fiyat alınamadı (maliyetle gösterildi)" der ve toplamın altında uyarı çıkar |
| `varlik-ekleme.png` | AC28 — `LINKUSD` eklenince "LINK-USD, kripto olarak okunacak" bilgisi yeniden çalıştırmadan sonra da okunabilir kalır |

**Canlı doğrulama (AP-05) hâlâ kullanıcı tarafından yapılacak:** gerçek `LINKUSD`/`HBARUSD` fiyatı ve gerçek
%40 kısmi satış, merge sonrası canlı uygulamada denenmeli; sonuç buraya eklenecek.
