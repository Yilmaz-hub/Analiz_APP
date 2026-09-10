# Kurtarma Rampaları (AP) Kataloğu

Bu katalog **kod çözümü** içindir. Her AP'nin **TAM promptunu Takım Yöneticisi çalıştırır.**
Bir AP tetiklendiğini düşünüyorsan: **kodunu söyle** (ör. "AP-02") ve yöneticiden promptu **iste**.
Kendi başına AP promptunu çalıştırma.

| Kod | Durum | Temel kural |
|-----|-------|-------------|
| AP-01 | derleme/çalışma zamanı hatası | önce teşhis, sonra TEK düzeltme; semptom susturma yasak |
| AP-02 | test kırmızı | önce karar: kod mu, test mi, spec mi; assert zayıflatma/silme/skip yasak |
| AP-03 | kararsız (flaky) test | deterministikleştir; skip/retry yasak; kanıt: art arda 5 yeşil |
| AP-04 | düzeltme regresyon yarattı | önce geri al, yeşile dön; dar düzeltme + kalıcı regresyon testi |
| AP-05 | "araştırılacak" bulgu | düzeltme değil MİNİMAL REPRO; repro yoksa gerekçeli kapanış |
| AP-06 | düzeltme tur sınırı aşıldı | DUR; kök neden spec'te mi planda mı — yalnız analiz, kod yok |
| AP-07 | plan dışına çıkıldı | sapma listesi; onaysız değişiklik geri alınır |
| AP-08 | iş ortasında spec değişti | önce spec güncellenir → delta plan → onay |
| AP-09 | bağlam bulanıklaştı | durum dosyası + oturum devri; kanıtsız "tamamlandı" yazılmaz |
| AP-10 | belirsizlik / docs-kod çelişkisi | VARSAYMA; seçenekleri bedelleriyle getir, yöneticiye sor |
| AP-11 | güvenli geri alma | git revert; history silme/force push yasak; migration risk raporu |
| AP-12 | performans hedefi tutmadı | önce ölç, tek optimizasyon, aynı yöntemle yeniden ölç |
