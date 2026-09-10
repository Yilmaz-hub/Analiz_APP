# Spec: 0003 — İşlem Kararları ve Performans Tutarlılığı (V1)

> Şablon: [TEMPLATE.md](TEMPLATE.md). Bitince → `specs/done/`.
> Rol: Analist — INTENT · CLARIFY · SPEC.
> Durum: Revizyon 19; Developer uygulaması ve testleri tamamlandı. Bağımsız QA ile commit/PR checkpoint'i bekliyor.
> Feature: Pozisyona göre açık işlem yönlendirmesi, ekran–geçmiş test–sanal işlem tutarlılığı ve güvenilir mum/işlem zamanlaması.

## Intent

AL uyarısıyla alıp SAT uyarısıyla satan kullanıcı, uyarıların kendi pozisyonu için ne anlama geldiğini açıkça bilmek ve gösterilen performansın uyguladığı işlem kurallarını temsil ettiğinden emin olmak istiyor.
V1, günlük mum esas alınarak kripto spot, BIST, altın ve Amerikan hisselerinde, kullanıcının kendi eklediği bu gruplardaki varlıklar dahil, elle gerçekleştirdiği kaldıraçsız alım ve sahip olunan varlığın satışını kapsar.
Onaylanan davranışa göre AL pozisyon yokken giriş, pozisyon varken tutma; BEKLE tek başına çıkış yaptırmadan tutma; SAT tam çıkış anlamına gelir; girişte konulan stop çalışırsa SAT beklenmez, kâr hedefinde otomatik satış yapılmaz ve stop yalnız kullanıcının teyit ettiği yükseltmeyle değişir.
İlk kapsam, önceki yol haritasının 1–3. adımlarıyla sınırlıdır: işlem anlamlarının netliği, ekran ile değerlendirmelerin tutarlılığı ve veri/işlem zamanlamasının doğruluğu.
Başarı, V1'de aynı koşullarda çelişmeyen yönlendirmeler ve maliyetleri açık değerlendirmeler elde etmektir; daha düşük kayıp, daha iyi net getiri ve daha az gereksiz işlem hedeflerinin performansla doğrulanması sonraki kapsamdadır.

## Requirements

- **R01 — Pozisyona göre eylem:** Kullanıcı piyasa yönüyle kendisinden beklenen eylemi ayırt edebilmeli; AL mevcut pozisyona ek alım anlamına gelmemeli.
- **R02 — BEKLE:** BEKLE tek başına mevcut pozisyonu kapatmamalı; pozisyon yokken yeni alım önermemeli.
- **R03 — Çıkış:** SAT mevcut pozisyonun tamamından çıkışı ifade etmeli; pozisyon yokken satış veya açığa satış önermemeli.
- **R04 — Koruyucu çıkış:** Zarar durdurma veya onaylı kâr koruma koşulu gerçekleştiğinde, AL veya BEKLE sürse bile pozisyonun tamamından çıkış belirtilmeli; gerekçesi görünmeli.
- **R05 — Gerçek pozisyon:** Bir uyarının oluşması, kullanıcının gerçek alım veya satış yaptığı anlamına gelmemeli; pozisyon bilgisi bilinmiyorsa bilinmediği açıkça belirtilmeli.
- **R06 — Ortak işlem anlamı:** Aynı başlangıç durumu, piyasa verisi, dönem, karar bileşenleri, ayarlar ve gerçekleşme varsayımları altında ekran, geçmiş test ve sanal takip aynı eylemi ifade etmeli.
- **R07 — Ortak işlem koşulları:** Kullanıcıya gösterilen giriş koşulları, stop, hedef ve yeniden giriş bekleme kuralları geçmiş test ile sanal takipte de geçerli olmalı; değerlendirmeler arasında bir bileşen eksikse birebir karşılaştırılabilirlik iddia edilmemeli.
- **R08 — Tamamlanmış veri:** Sinyal yalnız değerlendirme anında kapanışı doğrulanmış mumlara dayanmalı; tamamlanmış son mum kullanılabilir kalmalı.
- **R09 — Veri uygunluğu filtresi:** Eksik, geçersiz, güncelliğini yitirmiş veya kapanışı doğrulanamayan veri yeni alım yönlendirmesine dayanak olmamalı; engelleme nedeni görünmeli.
- **R10 — İşlem zamanlaması:** Sinyalin oluşma zamanı ile işlemin gerçekleşme zamanı ayırt edilmeli; geçmiş test ve sanal takip, kararın bilinebildiği andan önce işlem yapmış saymamalı.
- **R11 — Uygulanabilir fiyat:** Değerlendirmeler, onaylanan kullanıcı tepki süresini ve komisyon, alış–satış makası ile fiyat kayması varsayımlarını içermeli; varsayımlar kullanıcıya görünmeli.
- **R12 — Mum içi çıkış:** Değerlendirmeler stop ve hedef temaslarını yalnız kapanış fiyatına göre değerlendirmemeli; fiyat boşluğu ve aynı mumda iki seviyeye temas için onaylı ortak gerçekleşme kuralını uygulamalı.
- **R13 — Eksik bilgi:** Boş sonuç veya bilinmeyen maliyet, sıfır risk, sıfır maliyet ya da başarılı performans olarak sunulmamalı.
- **R14 — Güven görünürlüğü:** Gösterge uyum puanı, doğrulanmış kazanma olasılığı olarak sunulmamalı.
- **R15 — Pozisyon bilgisi:** Gerçek kullanıcı pozisyonu, geçmiş test pozisyonu ve sanal pozisyon birbirinden ayrı olmalı. Gerçek pozisyon bilinmiyor, yok veya açık olabilir; geçersiz bilgi ayrıca belirtilmeli, pozisyon yok kabul edilmemeli. Açık pozisyon için pozitif miktar, pozitif giriş fiyatı ve işlem zamanı ile kullanıcı teyidi gerekli olmalı.
- **R16 — Bilinmeyen/geçersiz pozisyon:** Pozisyon bilinmiyorsa sinyal bilgi olarak gösterilmeli, koşulsuz al/sat/tut eylemi verilmemeli. Pozisyon bilgisi geçersizse kişisel işlem eylemi ve koruyucu seviye üretilmemeli.
- **R17 — Teyit ve düzeltme:** Gerçek alım/satım durumu yalnız yön, miktar, gerçekleşen fiyat ve işlem zamanı içeren açık kullanıcı teyidiyle değişmeli; kullanıcı bu bilgileri ve stop yükseltmesinde yeni seviye/geçerlilik zamanını girmeyi kabul etmiştir. Kayıt zamanı ile gerçek işlem zamanı ayrı anlam taşır; sonradan girilen işlem kayıt anında yapılmış sayılmaz. Tekrarlanan teyit ikinci işlem yaratmamalı. Düzeltme yanlış bilgiyi düzeltmeli; iptal gerçekten gerçekleşmiş işlemi yapılmamış saymamalı.
- **R18 — Çakışan çıkışlar:** Geçerli koruyucu çıkış yön sinyalinden önce gelmeli ve aynı pozisyon için yalnız bir tam çıkış eylemi oluşmalı. Aynı mumda sırası belirlenemeyen stop/hedef temasında stop tetikleme önceliği kullanılmalı; bu öncelik gerçekleşme fiyatını belirlemez.
- **R19 — Korumanın geçerliliği:** Her koruma etkin veya devre dışı olarak ayırt edilmeli. İlk uzun pozisyon stopu pozitif ve girişin altında, etkin ilk hedef girişin üstünde olmalı. Sonradan güncellenen stop girişe eşit veya girişin üstünde olabilir; sırf bu nedenle geçersiz sayılmamalı. Devre dışı hedef stopu geçersiz kılmamalı. Geçersiz etkin koruma, geçerli koruma varmış gibi işlem değerlendirmesine alınmamalı.
  Günlük başlangıç stopu, giriş fiyatından 2,5 ATR çıkarılarak önerilir; ATR, alım kararına dayanak olan kapanmış günlük mumun değeridir. Ekran, geçmiş test ve sanal takip aynı giriş/karar/ATR girdileriyle aynı stopu hesaplar. Pozisyon açıldıktan sonra ATR değişimi ilk stopu kendiliğinden değiştirmez; aynı giriş kararına sonradan oluşmuş veya açık mumun ATR'si uygulanmaz. Kullanıcının kurumuna gerçekten koyduğunu teyit ettiği stop ile hesaplanan öneri farklıysa fark görünür olmalı; gerçek emir sessizce yeniden hesaplanmış sayılmamalı.
- **R20 — Yeniden giriş:** Zarar eden çıkıştan sonra, çıkışın gerçekleştiği günlük mum sayılmadan iki tamamlanmış günlük mum beklenir. Bu sırada AL bilgi olarak gösterilebilir, fakat yeni giriş önerilmez ve sanal yeni alım yapılmaz. İkinci mum tamamlanınca yalnız bekleme engeli kalkar; diğer giriş koşulları yeniden değerlendirilir. Kârlı çıkış bu beklemeyi başlatmaz. Takvim günleri yerine ilgili piyasanın tamamlanmış günlük mumları sayılır. Kullanıcının gerçek işlemleri bu tavsiyeden bağımsız teyit edilebilir; gerçekleşmiş bir işlem kayıt dışı bırakılmaz.
  Bekleme kararı, komisyon bilgisi mevcutsa net sonucun negatif olmasına dayanır. Komisyon bilinmiyorsa yalnız bekleme kararı için alış–satış farkı geçici olarak kullanılır; bu sınıflama görünür olmalı ve doğrulanmış net sonuç olarak sunulmamalıdır. Kullanılan sonucun tam sıfır olduğu başabaş çıkış bekleme başlatmaz. Sonradan komisyon girişi maliyet ve sonuç raporunu günceller; geçmiş sinyalleri ve o anda belirlenmiş bekleme kararını değiştirmez, yeni bir bekleme başlatmaz. Bu koruma devam eden ve sona ermiş beklemeler için geçerlidir; maliyet güncellemesi geriye dönük işlem veya sinyal yaratmaz.
- **R21 — Tetiklenme ve gerçekleşme:** Stop/hedefe tam eşit temas tetiklenmeye dahil olmalı. Temas, gerçekleşmiş işlem teyidi sayılmamalı. Kullanıcı tepkisi gerektiren uyarı ile kullanıcının kurumuna önceden verdiği koruyucu emir ayrı gerçekleşme varsayımları olarak ele alınmalı; AK02'deki onaylı ayrım kullanılmalı.
- **R22 — Bilinmeyen fiyat yolu:** Mum içi sıra veya tepki süresi sonrası fiyat belirlenemiyorsa sonuç onaylı varsayıma bağlı olarak belirtilmeli. Sonraki açılış fiyatı kendiliğinden muhafazakâr kabul edilmemeli. Girişten önceki fiyat teması pozisyon çıkışı sayılmamalı; temasın giriş sonrası olduğu belirlenemiyorsa kesin gerçekleşme iddia edilmemeli.
- **R23 — Maliyet ve nakit:** Bilinmeyen maliyet açıkça sıfır girilmiş maliyetten ayrılmalı. Negatif maliyet veya tepki süresi reddedilmeli. Her maliyetin hangi işlemde hangi fiyat üzerinden uygulandığı görünmeli; fiyata zaten dahil maliyet tekrar eklenmemeli. Aleyhte kayma, seçilirse, değerlendirme varsayımı olarak adlandırılmalı. Başlangıç nakdi ve miktar kuralı açık olmalı; ücret dahil kullanılabilir nakdi aşan giriş gerçekleşmiş sayılmamalı.
  Sanal alım miktarı, ilgili varlığın tanımlı geçerli adet/lot adımına aşağı yuvarlanır. Yuvarlama nedeniyle kullanılmayan tutar nakitte kalır; belirlenen işlem tutarı bütçe üst sınırıdır, her işlemde tam harcanması gerekmez. Bütçe en küçük geçerli miktara yetmiyorsa sanal giriş yapılmaz ve neden gösterilir. Geçerli miktar adımı bilinmiyorsa rastgele kesirli alım varsayılmaz; miktar uygunluğu doğrulanana kadar sanal giriş oluşturulmaz. Bu kural gerçek kullanıcının teyitli gerçekleşen miktarını yeniden yuvarlayarak değiştirmez.
  Sanal değerlendirme için makas ve fiyat kayması piyasa bazında sonradan girilebilir olmalı. Bilinmeyen makas/kayma açık sıfırdan ayrılmalı; ilgili sonuçta hangi etkinin hesaplanmadığı görünmeli ve eksik etkiler varken tam maliyet sonrası doğrulanmış sonuç iddia edilmemeli. Gerçek işlem hesabı kullanıcının teyit ettiği gerçekleşen fiyatı kullanmalı; o fiyata dahil makas veya fiyat kayması sanal varsayımla tekrar eklenmemeli. Ayrı komisyon giderinin kaydı bu fiyat farkından ayrı kalır. Sonradan seçilen sanal varsayımlar gerçek işlem fiyatını değiştirmemeli.
  Sanal strateji her varlık için bağımsız hesapla başlar: varsayılan başlangıç sermayesi 10.000 birim, yeni işlem tutarı komisyon hariç 1.000 birimdir; kullanıcı ikisini de değiştirebilir. Birim, ilgili varlığın fiyatlandığı para birimidir. Bilinen alış komisyonu alım tutarına ek olarak nakitten düşülür; toplam bedel kullanılabilir nakdi aşarsa sanal giriş gerçekleşmez. Komisyon bilinmiyorsa sıfır sayılmaz; maliyet ve ücret dahil nakit yeterliliği doğrulanmamış olarak gösterilir. Bir varlığın nakdi başka varlığın işlemini finanse etmez; bu bağımsız test hesapları tek gerçek portföy gibi sunulmaz. Farklı para birimlerindeki parasal sonuçlar doğrudan toplanmaz. Gerçek işlemler bu varsayılan tutarlara zorlanmaz; teyit edilen miktar ve fiyatla izlenir.
  Komisyon ilk kullanımda bilinmiyor bırakılabilmeli ve sonradan piyasa bazında varsayım veya gerçek işlem için ödenen tutar olarak girilebilmeli. Komisyon bilinmeden gerçek işlem teyidi engellenmemeli; maliyetin bilinmediği görünmeli ve doğrulanmış net sonuç sunulmamalı. Sonradan girilen komisyon ilgili maliyet/net sonuç hesabını güncellemeli, teyitli işlemin fiyatını, miktarını veya gerçekleşme zamanını değiştirmemeli. Genel komisyon varsayımı, işlem için açıkça girilmiş gerçekleşen komisyonun yerine sessizce geçmemeli. Sanal değerlendirmede ücret dahil nakit yeterliliği maliyet bilinmeden kesin doğrulanmış sayılmamalı.
- **R24 — Kullanılabilir geçmiş:** Güncellik son beklenen tamamlanmış muma göre değerlendirilmeli; normal kapalı seans tek başına eskilik oluşturmamalı. Açık son mum dışlanırken yeterli ve geçerli tamamlanmış geçmiş kullanılmalı. Asgari geçmiş, yalnız satır sayısıyla değil gerekli alanların hazır olması ve beklenen dönemlerin bulunmasıyla belirlenmeli; sıranın düzensiz olması tek başına veriyi kullanılamaz yapmamalı.
- **R25 — Hata ve açık pozisyon:** Veri yok, yetersiz/eski veri, geçersiz seçim, sağlayıcı, hesaplama ve sanal kayıt hatası birbirinden ayırt edilmeli; hata geçerli BEKLE veya boş başarı sayılmamalı. Açık pozisyonda veri doğrulanamıyorsa güncel riskin değerlendirilemediği belirtilmeli; koruma tetiklenmedi güvencesi verilmemeli. Son geçerli karar gösteriliyorsa zamanı ve güncel olmadığı görünmeli.
- **R26 — Karşılaştırma koşulları:** Eşitlik için piyasa, sembol, aralık, veri kesim anı ve revizyonu, başlangıç pozisyonu/nakdi, miktar kuralı, karar bileşenleri, iş kuralı ve ayar sürümü, maliyetler, gerçekleşme varsayımları ve hassasiyet aynı olmalı. Otorite mevcut modüllerden biri değil, onaylı spec davranışı olmalı. Görsel yuvarlama hesap farkını gizleyen eşitlik kanıtı sayılmamalı; farklı saat dilimi gösterimleri aynı gerçek zaman anını değiştirmemeli.
- **R27 — Eksik bileşen ve tekrar:** Zorunlu karar bileşeni eksikse birebir eşitlik başarısı gösterilmemeli; ayrı sonuç gösteriliyorsa eksik bileşen adı ve karşılaştırılamaz durumu belirtilmeli. Aynı kararın aynı koşullarda tekrar değerlendirilmesi ikinci sanal işlem yaratmamalı.
- **R28 — Görünürlük:** Geçerli duruma ait pozisyon belirsizliği, veri sorunu, hesaplanmamış net sonuç, çıkış gerekçesi ve puanın kazanma olasılığı olmadığı bilgisi kararın bulunduğu panelde ek etkileşim gerektirmeden görünmeli. Doğrulama kapsamı dışındaki mevcut seçenekler kaldırılmamalı; V1 tutarlılığının doğrulanmadığı belirtilmeli ve sessiz varsayılana dönüş yapılmamalı.
- **R29 — Kullanıcının çıkış biçimi:** V1'de kâr hedefi çıkışı ve otomatik iz süren stop devre dışıdır. Kullanıcı AL sırasında uygulamanın stop seviyesini esas alıp kurumuna stop emri koyar; ilk stop sabit kalır, yalnız kullanıcının yeni seviye ve geçerlilik zamanını teyit etmesiyle yükselir. Stop çalışmadıkça SAT beklenir; ayrı bir sayısal kâr eşiği veya ek trend çıkışı türetilmez. Hedef seviyesi bilgi olarak gösterilirse satış emri olmadığı belirtilir. Geçmiş test, zamanında kaydedilmemiş takdire bağlı stop yükseltmelerini sonradan tahmin etmez; böyle kayıt yoksa sabit stop + SAT varsayımı açıkça belirtilir. R18/R21'in etkin hedefe ilişkin genel hükümleri V1'de hedefi etkinleştirmez.
- **R30 — Seçici kullanım ve sonuç ayrımı:** Kullanıcı uygulamayı değişken saatlerde, dalgalanma arttığında daha sık kontrol eder ve her AL sinyalini uygulamaz. Gerçek pozisyon ve işlem sonucu yalnız teyit edilen işlemlerden hesaplanmalı; işlem yapılmayan AL gerçek alış sayılmamalı. Bütün uygun sinyalleri onaylı gerçekleşme varsayımlarıyla değerlendiren sanal strateji sonucu, kullanıcının gerçek işlem sonucundan ayrı adlandırılmalı ve onun kazanmış olacağı para olarak sunulmamalı. Kullanıcıya sabit kontrol saati veya sabit tepki süresi atanmamalı. Ortak koşul eşitliği, aynı girdilerle yapılan değerlendirmeler için geçerlidir; farklı gerçek/sanal işlem dizilerinin aynı getiriyi üretmesi beklenmez.

## Constraints

- **Kapsam sınırı:** Yalnız yol haritası adım 1–3; kripto spot, BIST, altın ve Amerikan hisseleri günlük mumla kapsam içidir. Kullanıcının bu gruplarda eklediği varlıklar da değerlendirilir; ETH/XAU örnekleri sabit sembol sınırı değildir. Veri kaynakları ve seans kuralları piyasa bazında tanımlanır; mevcut diğer zaman aralıkları kaldırılmaz.
- **İşlem sınırı:** Elle uygulama, kaldıraçsız alım ve tam pozisyon çıkışı; uyarı gerçek işlem teyidi sayılmaz.
- **Doğruluk hedefi:** Aynı onaylı koşulları kullanan karşılaştırma senaryolarında eylem, simüle işlem zamanı, fiyat ve net sonuç uyuşmazlığı sıfır olmalı. Gerçek kullanıcı işlemlerinin simülasyonla aynı fiyattan gerçekleşeceği taahhüt edilmez.
- **Performans hedefi:** V1 için getiri veya kazanma oranı taahhüdü yoktur. Kullanıcının ağ ve ilk çalışma dahil toplam beklemesi ile hesaplama süresi ayrı ölçülür; ekran ve geçmiş değerlendirme için ayrı bütçeler AK05'te kararlaştırılır. Yalnız ısınmış, ağ hariç ölçüm toplam kullanıcı deneyiminin kanıtı sayılmaz.
- **Finansal hassasiyet ve hata sunumu:** [Konvansiyonlar](../docs/conventions.md) geçerlidir; yuvarlama ve fiyat hassasiyeti onaylı AK02 kararına tabidir.
- **Yasaklar:** Yeni gösterge eklemek, daha yüksek geçmiş getiri için eşik/ağırlık optimizasyonu yapmak, ileride oluşacak veriyi geçmiş karara katmak, gerçek emir göndermek ve eksik kararları onaylı varsaymak V1 kapsamında yasaktır.
- **Bu feature için belge/yığın kararı:** Python/Streamlit ve mevcut Python test altyapısı geçerlidir. Mimari/alan belgeleri ile şablondaki .NET/React/xUnit atıflarının genel düzeltmesi ayrı dokümantasyon işidir; V1 teknoloji dönüşümü içermez (AK06).

### KAPSAM DIŞI (V1) — Sonraki adımlar unutulmamalı

- **Adım 4 — Güvenilir performans raporu:** Net getiri, sermayenin zirveden düşüşü, işlem başına beklenti, işlem sayısı; mevcut AL→SAT davranışı ve al-tut ile karşılaştırma; yükselen/düşen/yatay dönemler ve ayar seçiminde kullanılmamış dönemlerde değerlendirme.
- **Adım 5 — Çalkantıya uygun strateji denemeleri:** Mevcut filtrelerin etkisini ölçme; güçlü kırılım ve yükseliş içi geri çekilme girişlerini ayrı karşılaştırma; yeni stratejiyi önceden kararlaştırılmış ölçütlerle değerlendirme.
- **Adım 6 — Pozisyon riski ve kâr koruma geliştirmeleri:** Risk temelli pozisyon büyüklüğü, işlem/toplam portföy kayıp sınırları, sınır aşımında yeni alımı durdurma; sabit hedef, iz süren stop ve kademeli çıkış seçeneklerinin karşılaştırılması. V1'de mevcut koruyucu kuralların ortaklaştırılması kapsam içidir; yeni risk stratejileri kapsam dışıdır.
- **Adım 7 — Kesintisiz sanal doğrulama:** Sürekli ileri dönem takibi, sürüm ve veri kaynağıyla izlenebilirlik, sonradan oluşturulan kayıtları ayırma, yeterli işlem ve piyasa koşulu üzerinden başarı değerlendirmesi. V1'de mevcut sanal takibin davranış tutarlılığı kapsam içidir; kesintisiz işletim kapsam dışıdır.
- Otomatik gerçek emir, kaldıraç, açığa satış, ek alım ve kısmi satış.
- Günlük dışı aralıklar için V1 tutarlılık doğrulaması, stratejinin kârlılık/üstünlük doğrulaması, teknoloji dönüşümü ve genel arayüz yenilemesi. Kapsamdaki dört piyasa grubunun günlük davranış tutarlılığı V1'e dahildir.

## Context

- İş alanları: Karar paneli, sinyal değerlendirme, geçmiş işlem testi, sanal işlem takibi ve piyasa verisi uygunluğu.
- İncelenen mevcut parçalar: `app.py`, `signal_engine.py`, `technical_analysis.py`, `paper_trading.py`, `data_fetchers.py`, `config.py`; bunlar bağlam referansıdır, uygulama planı değildir.
- İş terimleri: AL = pozisyona göre giriş/tutma; BEKLE = yeni alım yok, tek başına çıkış yok; SAT = mevcut pozisyonun tamamından çıkış; koruyucu çıkış = yön sinyalinden bağımsız onaylı zarar/kâr koruma koşuluna bağlı çıkış.
- Önceki inceleme: BEKLE'nin geçmiş testte çıkış yaratması, ekran/test hedeflerinin ayrışması, sanal takibin mum içi stop temasını kaçırması ve son mumun koşulsuz dışlanması bu işin gerekçesidir; kullanıcı zararının miktarı bu incelemeyle ölçülmemiştir.
- Referanslar: [Rol](../docs/roles/analist.md), [alan belgesi](../docs/domain.md), [mimari](../docs/architecture.md), [test yaklaşımı](../docs/testing.md), [AP-10](../docs/ap.md).
- İlgili mevcut spec: [0001](0001-mobil-grafik-sadelestirme.md), yalnız doküman/yığın çelişkisi açısından bağlam; işlevsel bağımlılık yoktur.

### CLARIFY karar durumu

AK01–AK06 kapalıdır; AK05 sayısal bütçeleri 2026-09-10 ölçümü ve Takım Yöneticisi onayıyla kesinleşmiştir.

### AK01 — Kaynak ve ürün kimliği

**Onaylı karar:** Yahoo akışı ve XAU seçeneği korunsun. XAU, “GC=F vadeli altın referansı” olarak açıklansın; alınan gerçek ürünle aynı olduğu varsayılmasın. Varlığın piyasa, ürün, fiyatlama para birimi, veri kaynağı ve işlem kurumu ayrı bilgiler olsun. Özel sembolde otomatik eşleşme kesin değilse kimlik kullanıcı tarafından tamamlanabilsin. Kriptoda seçilmiş OKX/Binance önceliği korunsun; diğer gruplar Yahoo kullansın. Yedek kaynağa geçiş görünür olsun, farklı kaynakların mumları tek karar geçmişinde karıştırılmasın. Aynı kripto varlığında USD/USDT, AC104 uyarınca eşdeğerdir.

Türetilmiş gram-altın verisi bilgi amaçlı görünmeye devam etsin; gözlenmiş OHLC olmadığı için doğrulanmış mum içi stop simülasyonu ve ATR temelli işlem uygunluğu sunulmasın. GC=F veya gram-altın referansı farklı bir gerçek altın ürününün stop gerçekleşmesini teyit etmesin. Böyle bir gerçek işlem kullanıcı fiyatıyla kaydedilebilsin, ürün eşleşmesi bilinmiyor görünsün.

**Gerekçe:** Geniş izleme kapsamını korurken başka ürüne veya para birimine ait fiyatı gerçek işlem fiyatı gibi kullanmayı önler.

### AK02 — Stop, maliyet ve hassasiyet

**Onaylı karar:** Sanal modelde stop, işlem fiyatı OHLC'sine göre değerlendirilsin. Gerçek kurumun tetik dayanağı veya emir türü bilinmiyorsa “bilinmiyor” kalsın; bu bilgi sonradan girilebilsin. Kripto dışındaki gerçek işlemlere otomatik olarak OKX emir davranışı atanmasın. Uygulama gerçek stop gerçekleşmesini kullanıcı teyidi olmadan kaydetmesin.

- Önceden etkin stopun altında açılış varsa temel sanal satış fiyatı açılış; normal mum içi aşağı temas varsa stop seviyesi olsun. Eşitlik temas sayılsın. Bunlar maliyet öncesi model fiyatlarıdır, garanti edilen gerçekleşme değildir.
- Bekleyen SAT ile stop aynı açılışta çıkış gerektiriyorsa tek satış oluşsun. Açılış stopa eşit veya altındaysa stop gerekçesi öncelikli olsun.
- Yeni sanal alış açılışta gerçekleştikten sonra aynı mumda ilk stopa temas varsa stop çıkışı değerlendirilsin. Hesaplanan stop pozitif değilse alış oluşturulmasın.
- Mum ortasında yapılan gerçek stop yükseltmesinin öncesi/sonrası günlük OHLC ile ayrılamıyorsa yeni seviyeye geçmişte temas edilmiş gibi kesin sonuç üretilmesin. Gerçek teyit kaydı korunsun, tarihsel tetik sırası belirsiz gösterilsin.
- Miktar adımı, asgari miktar ve varsa asgari işlem tutarı seçilen işlem kurumunun ürün kurallarından doğrulansın. Kaynak bulunamıyorsa kullanıcı sonradan tanımlayabilsin; doğrulanmış sanal gerçekleşme uygunluk bilgisi tamamlanana kadar beklesin. Bütün hisselere veya bütün kriptoya tek adım atanmasın.
- Makas alanı tam alış–satış farkını baz puan olarak anlatsın; 100 baz puan = %1. Kayma alanı işlem başına olumsuz fiyat farkını aynı birimde anlatsın. Alışta temel fiyatın üzerine yarım makas ve kayma eklensin; satışta çıkarılsın. İkisi de aynı temel fiyat üzerinden toplansın. Komisyon çıkan işlem tutarına ayrıca uygulansın. Bu, açıkça etiketlenen bir simülasyon varsayımı olsun.
- Makas ve kayma sıfır dahil, negatif olmayan sonlu değerler olsun. Yarım makas ile kayma toplamı 10.000 baz puandan küçük olsun; sıfır/negatif gerçekleşme fiyatına yol açan giriş reddedilsin. Yüzdesel komisyon 0 dahil, %100 hariç aralıkta; teyitli mutlak ücret sıfır dahil negatif olmayan tutar olsun. Boş alan, açıkça girilmiş sıfırdan ayrı kalsın.
- Bilinmeyen makas/kaymada temel fiyatla brüt model gösterilebilsin; gerçekleşme ve net sonuç tam doğrulanmış sayılmasın. Gerçek teyitli fiyata makas/kayma ikinci kez eklenmesin.
- Sonradan değiştirilen sanal varsayımlar ileriye dönük geçerli olsun. Eski varsayımlarla kayıt korunarak ayrı bir yeniden hesaplama raporu üretilebilsin; yayımlanmış kararlar veya sanal defter sessizce yeniden yazılmasın. Önceden onaylanan geç komisyon düzeltmesi ve bekleme kuralları aynen korunsun.
- İşlem fiyatı, stop, tutar, ücret, bakiye ve getiri hesaplarında decimal kullanılsın. Miktar mevcut onaya göre aşağı yuvarlansın; önerilen satış stopu geçerli fiyat adımına yukarı yuvarlansın, alış fiyatına eşit/üstüne gelirse geçersiz sayılsın. Teyitli gerçek fiyat değişmesin. Para sunumu ilgili para birimi hassasiyetinde olsun; sunum yuvarlaması sonraki hesabın girdisi olmasın.
- Analitik ara hesap istisnası için [konvansiyonlar](../docs/conventions.md#v1-analitik-ara-hesap-istisnası) geçerlidir.

**Gerekçe:** Günlük verinin gösteremediği gerçekleşmeleri uydurmadan tekrarlanabilir sanal sonuç ve değişmeyen gerçek kayıt sağlar. Yalnız miktar adımı kontrolünün yanında kurumun asgari tutar şartını da dikkate alır.

### AK03 — Kapanış ve veri yeterliliği

**Onaylı karar:** Kripto günlük kararları UTC gün sınırında ortaklaştırılsın; OKX bu nedenle UTC günlük veri kullansın. Bunun eski OKX sinyallerini değiştirebileceği geçiş notunda açıkça belirtilecek. BIST, ABD hisseleri ve GC=F için ilgili ürünün seans/tatil takvimi ve saat dilimi korunsun; GC=F'ye hisse takvimi uygulanmasın.

- Sağlayıcının açık mum işareti varsa zaman geçmiş olsa da mum kapanmış sayılmasın. Teyit yoksa doğrulanmış seans sonu ve aşağıdaki yayın toleransı birlikte kullanılsın. Takvim/kimlik bilinmiyorsa kapanış doğrulanamıyor durumu gösterilsin.
- Yayın toleransı kriptoda 5 dakika, Yahoo günlük veride 30 dakika olsun. Bunlar sağlayıcının garanti ettiği gecikmeler değil, onaylı ürün eşikleridir. Yeni beklenen mum henüz gelmediyse tolerans sırasında önceki karar zamanı ile “yeni veri bekleniyor” görünsün; yeni alım üretilmesin. Toleransın tam sonunda eksiklik veri eskiliği sayılsın. Tatil/kapalı seans tek başına eskilik sayılmasın.
- En az 200 tamamlanmış, geçerli günlük mum istensin; ayrıca karar için kullanılan her kesitte göstergeler hazır olsun. Mevcut ML alt sınırı 150 ve kararlılık geçmişi 15 olduğu için 200 onaylı başlangıç sınırıdır; model kalitesi garantisi değildir.
- Trend, momentum, hacim, formasyon, ileri analiz, ML ve etkin rejim filtresi hazır olmalı. Geçerli bir hesaplamanın nötr çıkması kabul edilsin; hesaplanamaması nötr puana çevrilmesin. ML geçmiş değerlendirmede yalnız o tarihte bilinen veriyle üretilebilmeli; gelecekte eğitilmiş sonuç geçmişe uygulanmasın.
- Pozitif ve sonlu OHLC, tutarlı yüksek/düşük sıralaması ve geçerli hacim gerekli olsun. Sıfır hacimli gerçek mum otomatik silinmesin; gereken hacim göstergeleri üretilemiyorsa işlem kararı uygun sayılmasın. Eksik fiyat/hacim veya gösterge yapay varsayılanla doldurulmasın.
- Sırası bozuk mumlar zamanına göre düzenlenebilsin; birebir aynı tekrar tekilleştirilsin. Aynı zamanlı çelişkili mum, açıklanamayan beklenen seans boşluğu veya zorunlu alan eksikliği etkilenen karar geçmişini geçersiz kılsın. Resmen işlem olmayan gün eksik mum sayılmasın.
- Veri/bileşen sorunu yeni AL/SAT üretmesin. Son geçerli karar zamanı ile gösterilsin. Açık pozisyon ve teyitli stop görünür kalsın; güncel riskin değerlendirilemediği belirtilsin. Ayrı doğrulanmış stop teması varsa bildirim engellenmesin, gerçek satış teyidi yerine geçmesin.

**Gerekçe:** Kaynak saat farkından, açık mumdan veya başarısız göstergenin nötr sayılmasından doğan yanıltıcı kararları ayırır. 200 mum tek başına geçerlilik kanıtı olmaz.

- **AK04 — Ortak koruma kuralları (karara bağlı):** Günlük başlangıç stopu giriş fiyatının 2,5 ATR altında; ATR, alım kararına dayanak kapanmış günlük muma sabitlenir. İlk stop sabit, kâr hedefi satışı ve otomatik iz süren stop yok. Kullanıcı teyitli yükseltme geçerlilik zamanından itibaren uygulanır. Stop çalışmadıkça SAT beklenir; BEKLE tek başına satış değildir. Zarar sonrası çıkış mumu hariç iki tamamlanmış günlük mum beklenir. Komisyon biliniyorsa net sonuç, bilinmiyorsa geçici alış–satış farkı sınıflamada kullanılır; başabaş bekleme başlatmaz. Sonradan komisyon girilince maliyet ve sonuç raporu güncellenir; geçmiş sinyal ve belirlenmiş bekleme değişmez, yeniden bekleme başlatılmaz. Gerekçe: karar anında bilinmeyen bilgilerle yayımlanmış geçmişi yeniden yazmamak. Gerçekleşme fiyatı ve maliyet varsayımları AK02 kapsamında onaylıdır.

- **AK06 — Belge uyumu (Q25):** Karar: bu feature için Python/Streamlit ve mevcut Python test altyapısı geçerli; genel belge düzeltmesi ayrı iş. Gerekçe: gereksiz teknoloji dönüşümünü önlemek. Q24 görünürlük kararı R28'de tanımlıdır.

### AK05 — Performans: onaylı bütçeler

**Onaylı ölçüm yöntemi:** Tek varlık/500 günlük mum ekran hesabı, tek varlık/1.000 günlük mum geçmiş gerçekleşme hesabı ve ilk modül yükleme ayrı ölçülür. ML açık kalır. Beş bağımsız ilk yükleme ile üç hazırlık sonrası yirmi tekrar için medyan, p95 ve en uzun süre raporlanır; ağ hataları ayrıca görünür kalır.

**Onaylı bütçeler:** 500 mumluk ilk günlük sinyal hesabı yerel referans ortamında 2 saniyeyi; ısınmış karar paneli medyanı 0,01 saniyeyi ve en uzun tekrar 0,05 saniyeyi; ilk veri modülü yüklemesi toplam 8 saniyeyi aşmaz. Yerel Python 3.14.2 ölçümünde ilk sinyal 0,4385 saniye, ısınmış sinyal medyan/p95/en uzun 0,001017/0,001910/0,002680 saniye, beş ilk yüklemenin en uzunu 1,8264 saniye ve 1.000 mumluk gerçekleşme hesabı 0,0996 saniyedir. Ölçümde ağ çağrısı yapılmamıştır; ağ/sağlayıcı gecikmesi AC70 ile ayrı hata olarak sunulur.

**Gerekçe:** Bütçeler, ilk çalışma ile önbellekli tekrarı birbirinden ayırır ve daha önce 120 saniyeyi aşan üçüncü taraf gösterge yükleme gecikmesinin geri gelmesini engeller.

### Uygulama hazırlığı kararları

Aktif feature 0003 olarak yeniden numaralanır; bitmiş 0002 korunur. Eski kayıtlar korunarak geçiş kopyada doğrulanır. Eski sanal defter sanal kalır; eski portföy kayıtları kullanıcı teyidine kadar eski/teyitsiz görünür. Otomatik TP/SL kapanışları gerçek satış kanıtı veya doğrulanmış gerçek performans sayılmaz. Kullanıcı kayıtları sonradan tamamlayabilir.

## Acceptance Criteria
- [ ] **AC14 — Net sonuç eşitliği:** Aynı işlem dizisi ve AK02 maliyetleri altında geçmiş test ile sanal takip aynı net sonucu üretir.
- [ ] **AC15 — Bileşen farkının görünürlüğü:** Karşılaştırılan değerlendirmelerden birinde karar bileşeni eksikse kullanıcıya sonuçların birebir karşılaştırılamayacağı gösterilir.
- [ ] **AC16 — Açık mum:** Diğer bütün veriler sabitken yalnız henüz kapanmamış mumun fiyatları değiştirilirse yayımlanan kapanışa dayalı karar değişmez.
- [ ] **AC17 — Kapanış sınırı:** AK03'te seçilen kapanış doğrulama koşulu (varsa onaylı tolerans dahil) tam sağlandığında tamamlanmış son mum değerlendirmeye alınır.
- [ ] **AC18 — Gereksiz gecikme yok:** Gelen verinin son mumu zaten tamamlanmışsa sırf son sırada olduğu için dışlanmaz.
- [ ] **AC19 — Boş durum:** Hiç piyasa verisi yoksa işlem eylemi yerine veri yok durumu gösterilir.
- [ ] **AC20 — Asgari geçmiş alt sınırı:** AK03'te onaylanan asgari tamamlanmış mum sayısından bir eksik veri varsa yeni alım engellenir.
- [ ] **AC26 — Geçersiz maliyet:** Negatif komisyon girildiğinde değerlendirme başlatılmadan anlaşılır bir doğrulama mesajı gösterilir.
- [ ] **AC27 — Bilinmeyen maliyet:** Komisyon bilgisi verilmemişse sonuç maliyet sonrası doğrulanmış net performans olarak sunulmaz.
- [ ] **AC28 — Zaman sırası:** Hiçbir simüle işlem, dayandığı sinyalin bilinebildiği andan önce gerçekleşmiş gösterilmez.
- [ ] **AC29 — Sanal giriş/çıkış zamanı:** Kapanmış günlük mumdan oluşan uygun AL veya SAT, sonraki işlem gününün ilk günlük mum açılışında, o açılış fiyatı temel alınarak simüle edilir; sinyal mumunun kapanışından işlem yapılmış sayılmaz.
- [ ] **AC30 — Mum içi stop teması:** Açık pozisyonda önceden geçerli stopa mum içinde temas edilmişse kapanış stop üstünde olsa da stop çıkışı değerlendirilir.
- [ ] **AC31 — Stop ve bilgi hedefi teması:** V1'de aynı mumda stop ve bilgi amaçlı hedefe temas edildiğinde hedef satış yaratmaz; stop AK02'deki emir modeline göre değerlendirilir.
- [ ] **AC32 — Fiyat boşluğu:** Mum stopun ötesinde açıldığında stop çıkış fiyatı AK02'de onaylanan fiyat boşluğu kuralına uyar.
- [ ] **AC33 — Çıkış gerekçesi görünürlüğü:** Koruyucu çıkış eylemi gösterildiğinde çıkışı tetikleyen gerekçe aynı karar görünümünde yer alır.
- [ ] **AC34 — Güven puanı görünürlüğü:** Gösterge uyum puanı gösterildiğinde bunun kazanma olasılığı olmadığı aynı görünümde açıkça belirtilir.
- [ ] **AC35 — Varsayım görünürlüğü:** Bir performans değerlendirmesi gösterildiğinde kullanılan maliyet ve gerçekleşme varsayımları kullanıcı tarafından görülebilir.
- [ ] **AC36 — Hesaplama süresi hedefi:** AK05'te onaylanan referans koşullarda hesaplama süresi, ilgili ekran veya geçmiş değerlendirme bütçesini aşmaz.
- [ ] **AC37 — Bilinmeyen pozisyonda eylem:** Pozisyon bilinmiyorken AL gösterilirse koşulsuz yeni alım talimatı verilmez.
- [ ] **AC38 — Geçersiz pozisyon:** Açık pozisyon için sıfır miktar girildiğinde geçersiz pozisyon bilgisi gösterilir; durum pozisyon yok olarak sunulmaz.
- [ ] **AC39 — Geçersiz pozisyonda koruma:** Açık pozisyonun giriş fiyatı eksikse kişisel stop seviyesi üretilmez.
- [ ] **AC40 — Satış teyidi:** Kullanıcı gerçek satış teyidi vermeden SAT uyarısı oluştuğunda gerçek pozisyon kapanmış sayılmaz.
- [ ] **AC41 — Tekrarlanan teyit:** Aynı gerçek işlem teyidi yeniden verildiğinde ikinci işlem oluşmaz.
- [ ] **AC42 — Düzeltme:** Gerçekleşmiş işlemin yanlış fiyatı düzeltildiğinde işlem gerçekleşmemiş sayılmaz.
- [ ] **AC43 — Pozisyon ayrımı:** Sanal takipte bir pozisyon açılması kullanıcının gerçek pozisyonunu değiştirmez.
- [ ] **AC44 — Tek çıkış:** Aynı açık pozisyonda koruyucu çıkış ve SAT birlikte oluştuğunda yalnız bir tam çıkış eylemi oluşur.
- [ ] **AC45 — Güncellenmiş stop:** Kullanıcının teyit ederek giriş üstüne yükselttiği geçerli stop, sırf girişin üstünde olduğu için reddedilmez.
- [ ] **AC46 — Bağımsız koruma:** Hedef devre dışıyken geçerli stop etkin kalır.
- [ ] **AC47 — Geçersiz ilk stop:** Yeni uzun pozisyon için ilk stop girişe eşit veya girişin üstündeyse koruma doğrulama hatası gösterilir.
- [ ] **AC48 — Bekleme alt sınırı:** Zarar sonrası çıkış mumu hariç sıfır veya bir günlük mum tamamlanmışken yeni alım önerilmez ve sanal giriş oluşmaz.
- [ ] **AC49 — Bekleme tam sınırı:** Zarar sonrası çıkış mumu hariç ikinci günlük mum tamamlandığında bekleme engeli kalkar; diğer giriş koşulları sağlanırsa sanal işlem AK02'deki sonraki günlük açılış kuralını izler.
- [ ] **AC50 — Bekleme kapsamı:** Net kârla kapanan işlem tek başına zarar sonrası yeniden giriş beklemesini başlatmaz.
- [ ] **AC51 — Stop eşit temas:** Önceden etkin stopa fiyat tam eşit olduğunda stop tetik koşulu sağlanır.
- [ ] **AC52 — Bilgi hedefi eşit temas:** V1'de fiyat bilgi amaçlı hedefe tam eşit olduğunda hedef kaynaklı satış oluşmaz.
- [ ] **AC53 — Bilgi hedefi boşluğu:** V1'de bilgi amaçlı hedefin üstünde açılış tek başına satış oluşturmaz.
- [ ] **AC54 — Belirsiz sıra:** Giriş ile mum içi stop temasının sırası bilinmiyorsa sonuç kesin gözlenmiş gerçekleşme olarak sunulmaz.
- [ ] **AC55 — Giriş öncesi temas:** Stop teması girişten önce gerçekleşmişse o temas açık pozisyonun çıkışı sayılmaz.
- [ ] **AC56 — Açık sıfır maliyet:** Kullanıcı komisyonu açıkça sıfır belirttiğinde komisyon bilinmiyor olarak gösterilmez.
- [ ] **AC57 — Çifte maliyet:** Kullanılan fiyata makasın dahil olduğu belirtilmişse aynı makas ikinci kez maliyet olarak eklenmez.
- [ ] **AC58 — Negatif tepki süresi:** Negatif tepki süresi girildiğinde gerçekleşme değerlendirmesi doğrulama mesajıyla engellenir.
- [ ] **AC59 — Nakit sınırı:** Giriş bedeli ve ücret toplamı kullanılabilir nakdi aştığında giriş gerçekleşmiş sayılmaz.
- [ ] **AC60 — Kapalı seans:** Son beklenen tamamlanmış mum mevcutsa normal seans kapanışı tek başına eski veri hatası oluşturmaz.
- [ ] **AC61 — Açık mumla geçerli geçmiş:** Son mum açıkken yeterli ve geçerli tamamlanmış geçmiş varsa açık mumun varlığı tek başına değerlendirmeyi engellemez.
- [ ] **AC62 — Isınma eksikliği:** Mum sayısı yeterli olsa bile AK03'te zorunlu alanlar hazır değilse yeni alım üretilmez.
- [ ] **AC63 — Çelişkili fiyat aralığı:** Karara esas mumun en yüksek fiyatı en düşük fiyatından küçükse bu mumdan yeni karar üretilmez.
- [ ] **AC64 — Açık pozisyonda veri sorunu:** Açık pozisyonun güncel verisi geçersizse güncel riskin doğrulanamadığı karar panelinde gösterilir.
- [ ] **AC65 — Eski karar görünümü:** Veri sorunu sırasında son geçerli karar gösteriliyorsa kararın zamanı ve güncel olmadığı bilgisi görünür.
- [ ] **AC66 — Eksik bileşen:** Zorunlu karar bileşeni eksikken birebir eşitlik başarısı gösterilmez.
- [ ] **AC67 — Hesap eşitliği:** Ekranda aynı yuvarlanmış değeri gösteren fakat onaylı hesap hassasiyetinde farklı iki net sonuç eşit kabul edilmez.
- [ ] **AC68 — Aynı zaman:** Aynı gerçek zaman anının farklı saat dilimi gösterimleri zaman uyuşmazlığı sayılmaz.
- [ ] **AC69 — Sanal tekrar:** Aynı başlangıç durumu ve veri kesimiyle aynı karar yeniden değerlendirildiğinde ikinci sanal işlem oluşmaz.
- [ ] **AC70 — Sistem hatası:** Sağlayıcı zaman aşımı gerçekleştiğinde boş başarılı değerlendirme gösterilmez.
- [ ] **AC71 — Kapsam görünürlüğü:** V1 doğrulama kapsamı dışındaki mevcut seçenek seçildiğinde V1 tutarlılığının doğrulanmadığı gösterilir.
- [ ] **AC72 — Mesaj erişimi:** Pozisyon bilinmiyor bilgisi karar panelinde ek tıklama veya imleç hareketi gerektirmeden görünür.
- [ ] **AC73 — Toplam bekleme:** AK05 referans koşullarında ağ ve ilk çalışma dahil kullanıcı bekleme süresi ilgili toplam süre bütçesini aşmaz.
- [ ] **AC74 — Sabit stop:** Kullanıcı stop değişikliği teyit etmedikçe fiyat yükselişi veya ekranın yeniden değerlendirilmesi pozisyonun geçerli stopunu değiştirmez.
- [ ] **AC75 — Stop yükseltme zamanı:** Teyitli stop yükseltmesi, kullanıcının belirttiği geçerlilik anından önceki fiyat temaslarına uygulanmaz.
- [ ] **AC76 — Kaydı olmayan takdir:** Geçmiş stop yükseltme kaydı olmayan değerlendirmede kullanıcı takdiriyle yükseltme yapılmış sayılmaz; sabit stop + SAT varsayımı görünür.
- [ ] **AC77 — Uygulanmayan sinyal:** AL sinyaline rağmen kullanıcı alış teyidi vermediğinde gerçek portföye işlem eklenmez.
- [ ] **AC78 — Sonradan işlem girişi:** Kullanıcı işlemi sonradan kaydettiğinde gerçek işlem zamanı olarak teyit ettiği zaman korunur; kayıt zamanı bunun yerine geçmez.
- [ ] **AC79 — Gerçek/sanal sonuç görünürlüğü:** Gerçek ve sanal strateji sonuçları birlikte gösterildiğinde hangi sonucun teyitli kullanıcı işlemlerinden, hangisinin simülasyondan geldiği açıkça belirtilir.
- [ ] **AC80 — Komisyonsuz teyit:** Komisyon bilinmiyorken gerçek işlem fiyat, miktar ve zamanla kaydedilebilir; bu işlem için doğrulanmış net sonuç gösterilmez.
- [ ] **AC81 — Sonradan komisyon:** Teyitli işleme komisyon sonradan eklendiğinde maliyet hesabı güncellenir; kayıtlı işlem fiyatı, miktarı ve gerçekleşme zamanı aynı kalır.
- [ ] **AC82 — Sonraki mum yok:** Sinyalden sonraki günlük mum henüz mevcut değilse sanal AL/SAT gerçekleşmiş sayılmaz; bekleyen sinyal olarak gösterilir.
- [ ] **AC83 — Başlangıç stop hesabı:** Aynı giriş fiyatı ve geçerli ATR ile ekran, geçmiş test ve sanal takip başlangıç stopunu giriş fiyatı eksi 2,5 ATR olarak hesaplar; örneğin giriş 100 ve ATR 4 iken hassasiyet işlemi öncesi stop 90 olur.
- [ ] **AC84 — ATR referansının korunması:** Alım kararına dayanak kapanmış günlük mumun ATR'si 4 ve giriş fiyatı 100 iken, sonraki mumun ATR'si 6 olsa bile aynı giriş kararının başlangıç stopu hassasiyet işlemi öncesi 90 kalır.
- [ ] **AC85 — Bilinen komisyonla net zarar:** Alış–satış farkı pozitif olsa bile bilinen komisyon dahil net sonuç negatifse zarar sonrası bekleme başlar.
- [ ] **AC86 — Bilinmeyen komisyonla geçici sınıf:** Komisyon bilinmiyorsa bekleme kararı alış–satış farkının negatif olup olmamasına göre verilir ve bu sınıflamanın geçici olduğu görünür; doğrulanmış net sonuç gösterilmez.
- [ ] **AC87 — Başabaş çıkış:** Bekleme için kullanılan net veya geçici alış–satış farkı tam sıfırsa çıkış bekleme başlatmaz.
- [ ] **AC88 — Geç komisyonla yeni bekleme yok:** Başta bekleme başlatmayan çıkış, komisyon sonradan girilince raporda zarara dönse bile yeni bekleme veya geriye dönük sinyal oluşturmaz.
- [ ] **AC89 — Geç komisyonla mevcut bekleme korunur:** Bekleme sürerken veya bittikten sonra komisyon bilgisi güncellendiğinde daha önce belirlenmiş beklemenin başlangıç ve bitiş sınırları değişmez.
- [ ] **AC90 — Sanal varsayılanlar:** Kullanıcı farklı ayar vermediğinde her varlığın sanal başlangıç sermayesi 10.000, işlem tutarı 1.000 olarak ve fiyatlama para birimiyle gösterilir.
- [ ] **AC91 — Değiştirilebilir sanal tutarlar:** Kullanıcı başlangıç sermayesini ve işlem tutarını geçerli farklı değerlerle seçtiğinde yeni değerlendirme bu değerleri kullanır; 10.000/1.000 varsayılanına sessizce dönmez.
- [ ] **AC92 — Bağımsız varlık hesabı:** Bir varlığın sanal işleminde nakit değişmesi başka varlığın sanal bakiyesini değiştirmez.
- [ ] **AC93 — Para birimi ayrımı:** TL, USD ve USDT cinsinden parasal sonuçlar birlikte gösterildiğinde bu tutarlar doğrudan tek toplam olarak sunulmaz.
- [ ] **AC94 — Komisyon hariç işlem tutarı:** Miktar yuvarlaması gerektirmeyen örnekte 10.000 nakit, 1.000 alım tutarı ve bilinen 2 birim alış komisyonuyla sanal alışın ardından nakit 8.998 olur; alım tutarı komisyon nedeniyle 998'e indirilmez.
- [ ] **AC95 — Ücret dahil nakit sınırı:** Miktar yuvarlaması gerektirmeyen örnekte 1.000 alım tutarı ve 2 birim komisyon için nakit tam 1.002 ise nakit kontrolü geçer; 1.002'nin altındaysa sanal giriş oluşmaz.
- [ ] **AC96 — Bilinmeyen ücretle nakit görünürlüğü:** Komisyon bilinmiyorken sanal değerlendirmede ücret dahil nakit yeterliliği doğrulanmış olarak gösterilmez.
- [ ] **AC97 — Bilinmeyen makas/kayma:** Sanal sonuçta makas veya kayma bilinmiyorsa ilgili etkinin hesaplanmadığı gösterilir; bilinmeyen değer sıfır maliyet olarak sunulmaz.
- [ ] **AC98 — Sonradan piyasa maliyeti:** Kullanıcı bir piyasa için geçerli makas/kayma varsayımı girdiğinde o ayarla başlatılan sanal değerlendirme seçilen değerleri kullanır; başka piyasanın varsayımı değişmez.
- [ ] **AC99 — Gerçek fiyata ek kayma yok:** Teyitli gerçek işlem fiyatına sanal makas/kayma ayarı tekrar uygulanmaz; bu ayarı değiştirmek gerçek işlemin fiyat farkından doğan sonucunu değiştirmez.
- [ ] **AC100 — Miktarı aşağı yuvarlama:** Sanal bütçe 1.000, fiyat 300 ve geçerli miktar adımı 1 iken miktar 3 olur; komisyon hariç 900 harcanır ve bütçenin kullanılmayan 100 birimi nakitte kalır.
- [ ] **AC101 — Asgari miktara yetmeyen bütçe:** Sanal bütçe en küçük geçerli alım miktarının bedelinden düşükse giriş oluşturulmaz ve yetersiz tutar nedeni görünür.
- [ ] **AC102 — Bilinmeyen miktar adımı:** Geçerli adet/lot adımı bilinmiyorsa kesirli miktar uydurulmaz; sanal giriş engeli ve nedeni görünür.
- [ ] **AC103 — Gerçek miktarı koruma:** Kullanıcının teyit ettiği gerçek işlem miktarı, sanal miktar yuvarlama ayarı nedeniyle değiştirilmez.

### Onaylı düzeltme — USD/USDT eşdeğerliği

Aynı kripto varlığının USD ve USDT fiyatlamaları V1 sinyal ve stop değerlendirmesinde birebir eşdeğer kabul edilir (1 USD = 1 USDT model varsayımı). ETH/USD ve ETH/USDT gibi çiftler için yalnız bu farktan kaynaklanan eşleşme engeli veya kullanıcı teyidi istenmez. Kaynak fiyatları değiştirilmez; gerçek işlemde kullanıcının teyit ettiği fiyat ve para birimi korunur. Bu karar AC93'teki parasal sonuçları doğrudan toplamama kuralını değiştirmez.

- [ ] **AC104 — USD/USDT eşdeğerliği:** Diğer bütün uygunluk koşulları sağlandığında ETH/USD kaynak verisi ETH/USDT pozisyonunun sinyal ve stop değerlendirmesinde kullanılabilir; yalnız USD/USDT farkı değerlendirmeyi engellemez veya ek eşleştirme teyidi gerektirmez.

- [ ] **AC105 — Altın referansı:** XAU görünümünde GC=F vadeli altın referansı olduğu görünür; farklı gerçek altın ürününde gerçekleşmiş stop teyidi sayılmaz.
- [ ] **AC106 — Türetilmiş gram altın:** Türetilmiş GRAM_TRY verisinde stop simülasyonu ve ATR işlem uygunluğu doğrulanmış olarak sunulmaz.
- [ ] **AC107 — Yedek kaynak:** Yedek kaynak kullanıldığında değişim görünür; tek karar geçmişinde iki kaynağın mumları karıştırılmaz.
- [ ] **AC108 — Açılışta çıkış çakışması:** Bekleyen SAT varken açılış etkin stopa eşit veya altındaysa tek sanal satış oluşur; gerekçe stop, maliyet öncesi fiyat açılıştır.
- [ ] **AC109 — Alış mumunda stop:** Açılışta gerçekleşen sanal alışın pozitif ilk stopuna aynı mumda aşağı temas olduğunda stop seviyesinden maliyet öncesi tek çıkış oluşur.
- [ ] **AC110 — Pozitif olmayan stop:** Giriş eksi 2,5 karar ATR sonucu sıfır veya negatifse sanal alış oluşmaz.
- [ ] **AC111 — Maliyet formülü:** Temel fiyat 100, tam makas 100 baz puan ve kayma 25 baz puanken komisyon öncesi model alış fiyatı 100,75 ve satış fiyatı 99,25 olur.
- [ ] **AC112 — Maliyet üst sınırı:** Yarım makas ve kayma toplamı 10.000 baz puanken giriş reddedilir; 9.999 baz puan diğer koşullar geçerliyken bu üst sınırdan engellenmez.
- [ ] **AC113 — Komisyon sınırı:** Yüzdesel komisyon için sıfır kabul edilir; negatif, sonlu olmayan, yüzde 100 veya üstü değer mesajla reddedilir.
- [ ] **AC114 — Varsayım revizyonu:** Sanal makas/kayma değiştiğinde önceki sanal kayıt korunur; sonraki değerlendirme yeni varsayımı kullanır.
- [ ] **AC115 — Stop fiyat adımı:** Ham ilk stop 90,03 ve fiyat adımı 0,05 iken öneri 90,05 olur; öneri girişe eşit veya üstündeyse ilk stop geçersizdir.
- [ ] **AC116 — Kripto yayın sınırı:** Beklenen UTC günlük mum eksikken kapanıştan 4 dakika 59 saniye sonra yeni veri bekleniyor, tam 5 dakika sonra eski veri görünür; iki durumda da yeni AL oluşmaz.
- [ ] **AC117 — Yahoo yayın sınırı:** Beklenen Yahoo günlük mum eksikken seans kapanışından 29 dakika 59 saniye sonra yeni veri bekleniyor, tam 30 dakika sonra eski veri görünür; iki durumda da yeni AL oluşmaz.
- [ ] **AC118 — Asgari geçmiş:** Diğer uygunluk koşulları sağlandığında 199 geçerli kapanmış günlük mum alımı engeller; 200 mum asgari geçmiş filtresini geçer.
- [ ] **AC119 — Açık mum işareti:** Sağlayıcının açık işaretli mumu, seans sonu ve tolerans geçse dahi sinyal geçmişine alınmaz.
- [ ] **AC120 — Bileşen hatası:** Zorunlu bileşen hesaplanamadığında yeni AL/SAT oluşmaz ve bileşen adı görünür; geçerli nötr sonuç hata sayılmaz.
- [ ] **AC121 — ML zaman tutarlılığı:** Yalnız değerlendirme anından sonraki veri değiştirildiğinde geçmiş kesite ait ML kararı değişmez.
- [ ] **AC122 — Eski teyitsiz kayıt:** Eski otomatik kapanış kullanıcı satış teyidi olmadan doğrulanmış gerçek performansa katılmaz; eski/teyitsiz olarak görünür.

## Definition of Done

- [ ] AK05 sayısal süre hedefleri onaylandı; tüm onaylı AK kararları ve bağımlı kriterler uygulandı.
- [ ] Her kabul kriteri ayrı test/kanıtla karşılandı; [test yaklaşımı](../docs/testing.md) uygulandı.
- [ ] Kullanıcıya görünen kriterler için ekran görüntüleri sağlandı.
- [ ] Kritik akışın uçtan uca kanıtı sağlandı: pozisyon yok → AL → gerçek alım teyidi → BEKLE ile tutma → koruyucu çıkış veya SAT.
- [ ] Pipeline kontrolleri ve PR gereklilikleri [git kurallarına](../docs/git.md) göre tamamlandı.
- [ ] Para/yüzde ve hata sunumu [konvansiyonlara](../docs/conventions.md) göre doğrulandı.
- [ ] Bağımsız QA değerlendirmesi ve Takım Yöneticisi kararı alındı; Analist kendi üretimini QA denetimi olarak onaylamadı.
- [ ] SCORECARD gerçek kanıtlarla güncellendi; kapanışta spec `specs/done/` altına taşındı.

---

## SCORECARD

| Metrik | Değer |
|--------|-------|
| Spec revizyon sayısı | 19 — AK05 bütçeleri ve 0003 uygulama kanıtı işlendi |
| Düzeltme turu sayısı | 0 — bağımsız QA henüz başlamadı |
| Bulgu gerçek/gürültü oranı | QA değerlendirmesi yapılmadı |
| Regresyon sayısı | Developer doğrulamasında 0; 291/291 test geçti |
| Kaçan hata | Ölçülmedi |
