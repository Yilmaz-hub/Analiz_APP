# Spec: 0010 — Pozisyon Adı ile Varlık Listesi Eşleşmesi

> Şablon: [TEMPLATE.md](TEMPLATE.md). Rol: Analist/Developer (acil). Revizyon 1 (2026-10-05).
> Durum: **TASLAK — acil düzeltme.** Kaynak: Takım Yöneticisi'nin yayındaki bildirimi: 0009 sonrası portföy yükleniyor ama `link` ve `Hbar` için "varlık listesinde bulunamadı (varlık adı değişmiş ya da silinmiş olabilir)" yazıyor; kâr/zarar hesaplanamıyor.

## Kök neden
Pozisyon kaydındaki `Coin` adı, varlık listesindeki anahtarla **birebir** eşleşmediğinde sembol bulunamıyor ve fiyat hiç sorulmuyor. Gerçek kayıt verisi bu oturumdan görülemiyor; adlar büyük/küçük harf, boşluk ya da kısa ad (`link` ↔ `Chainlink (LINK)`) bakımından ayrışmış olabilir. Birebir eşleşme gerektiren tek yer fiyat yolu; diğer paneller seçili varlığın anahtarını kullanır.

## Intent
Pozisyon adı listedeki varlığa **güvenli** biçimde eşlenir; eşlenemeyen satır ayrıca ve açıkça belirtilir.

## Requirements
- **R01:** Eşleme sırası: birebir; büyük/küçük harf ve boşluktan bağımsız; adın listedeki tek bir varlığın adındaki sözcük ya da sembol tabanıyla örtüşmesi.
- **R02:** Birden fazla aday varsa ya da aday yoksa tahmin edilmez; satır "varlık listesinde bulunamadı" der.
- **R03:** Kayıtlar yeniden yazılmaz; eşleme yalnız okuma anında yapılır.

## Acceptance Criteria
- [ ] **AC01 — Büyük/küçük harf, R01:** `" hbar "` listedeki `Hbar`'a çözülür.
- [ ] **AC02 — Kısa ad, R01:** `link`, tek aday olan `Chainlink (LINK)`'e çözülür.
- [ ] **AC03 — Belirsiz/bilinmeyen, R02:** Çok aday ya da aday yok → `None`.
- [ ] **AC04 — Ekranda fiyat, R01/R03:** Adı listeden farklı yazılmış pozisyon tabloda "canlı" fiyatlanır, "bulunamadı" yazmaz.

## Definition of Done
- [ ] Testler yeşil; ayrı QA kabulü
- [ ] Yayında: `link`/`Hbar` fiyatlanır. Hâlâ "bulunamadı" ise ekranda görünen ad ile varlık listesi adları karşılaştırılır (veri gerekir).

## Bilinen sınır
Eşleme nedeni gerçek veriden doğrulanamadı; düzeltme olası ad farklarını kapsar. Ad tamamen başkaysa (silinmiş/yeniden adlandırılmış) kullanıcı varlığı listeye o adla eklemelidir.

## SCORECARD
| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 1 |
| Düzeltme turu sayısı | 0 |
| Bulgu gerçek/gürültü oranı | Ölçülmedi |
| Regresyon sayısı | 0 |
| Kaçan hata | 0009 testleri adı listeyle birebir eşleşen kayıtlar kullandı |
