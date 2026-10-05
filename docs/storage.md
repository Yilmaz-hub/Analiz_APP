# Kayıtların Saklanması

> Kaynak spec: [specs/0004-varlik-ve-pozisyon-kayitlarini-koruma.md](../specs/0004-varlik-ve-pozisyon-kayitlarini-koruma.md).
> Kod: [`storage.py`](../storage.py).

## Neden değişti

Kayıtlar eskiden göreli dosya adlarıyla (`portfolio.json`, `varliklar.json`)
yazılıyordu. Bunun iki sonucu vardı:

1. Kayıt yeri, uygulamayı başlatan **çalışma dizinine** bağlıydı; başka bir
   klasörden açılınca başka bir kayıt kümesi görünüyordu.
2. Bu dosyalar sürüm kontrolünde izlenmediği için **yayın her kurulduğunda
   yoktular**; uygulama koddaki yerleşik listeye ve boş portföye dönüyordu.
   Kullanıcının gördüğü "eklediğim varlıklar kayboluyor" davranışı buydu.

## Şimdi nasıl

Varlık listesi, portföy ve gerçek işlem günlüğü (`position_journal`, spec 0003 Q13),
`app_records` tablosunda birer **belge** olarak
tutulur (`doc_key`, `payload`, `updated_at`). Belge içeriği eskiden dosyaya
yazılan JSON'un aynısıdır; alan alan şemaya çevrilmez, böylece mevcut
para/yüzde gösterimi olduğu gibi korunur.

Yazma, tek satırlık `upsert` ile yapılır; atomikliği veritabanı garanti eder.

## Arka uçlar

Tek kod yolu, iki arka uç — ikisi de SQLAlchemy URL'i ile ayrışır:

| Ortam | URL kaynağı | Sonuç |
|---|---|---|
| Yerel | (yok) → `sqlite:///<veri klasörü>/analiz.db` | Çalışma dizininden bağımsız |
| Yayın | `ANALIZ_APP_DB_URL` ya da `st.secrets["db_url"]` | Yeniden yayınlamadan etkilenmez |

Öncelik: `ANALIZ_APP_DB_URL` > `st.secrets["db_url"]` > yerel SQLite.

**Yerel veri klasörü:** `ANALIZ_APP_DATA_DIR` verilmişse orası, yoksa
`~/.analiz_app`.

## Yayın kurulumu

Yayında kalıcılık, yalnız uygulama sürecinin **dışındaki** bir veritabanıyla
sağlanır. Yönetilen bir Postgres (ör. Neon, Supabase) açıp bağlantı dizesini
Streamlit secrets'a yazmak yeterlidir:

```toml
# .streamlit/secrets.toml  (yayında: uygulama ayarlarındaki Secrets alanı)
db_url = "postgresql+psycopg://kullanici:parola@host/veritabani?sslmode=require"
```

Neon'un panelden verdiği `postgresql://…` ve `postgres://…` adresleri olduğu gibi yapıştırılabilir:
uygulama bunları kurulu sürücüye (`postgresql+psycopg://…`) kendiliğinden çevirir; çevreleyen boşluk ve tırnak
atılır, sürücüsü açıkça yazılmış adres ve SQLite değişmez (`storage.normalize_db_url`, spec 0007 R01).
Bağlanılamazsa kenar çubuğu "kayıtlarınız silinmedi, bağlantı adresini kontrol edin" der.

Sürücü `requirements.txt` içindedir (`psycopg[binary]`). Tablo ilk açılışta
kendiliğinden oluşur.

> Bu ayar yapılmadan yayında **AC19** (yeniden yayınlama sonrası koruma) ve
> **AC05b** (uyku sonrası koruma) sağlanamaz.

## Eski kayıtların içeri alınması

Depoda karşılığı olmayan bir belge varsa, uygulama açılışında eski dosyalar
çalışma dizini, proje klasörü ve masaüstünde aranır; bulunan dosya
**kopyalanarak** içeri alınır. Kaynak dosyaya dokunulmaz ve depodaki mevcut
kayıtların üzerine yazılmaz.

## Erişim sorunu

Okuma başarısız olursa `StorageAccessError` yükselir. Uygulama **varsayılan
listeye düşmez**: kullanıcıya erişim sorunu bildirilir ve kayıt değiştiren
tüm kontroller gizlenir. Sessizce varsayılana düşmek, bir sonraki yazmada
gerçek kayıtların üzerine yazılmasına yol açıyordu — kaybın kendisi buydu.

## İşlem günlüğü

`PositionJournal` kalıcı depoya yazar; yerel `position_journal.json` yalnız bir kereliğine içeri alınır (depoda belge yoksa). Okuma ya da çözümleme başarısız olursa `StorageAccessError` yükselir ve **boş günlükle devam edilmez**; arayüz işlem teyidini kapatır, kayıt silinmez. Yazma başarısızsa bellekteki değişiklik geri alınır.

## İşlem varsayımları

Sermaye, işlem tutarı, miktar adımı, makas, kayma ve komisyon `trade_settings` belgesinde varlık başına tutulur (`trade_settings.py`). Geçmiş test ve sanal takip aynı kaydı okur. Boş alan *bilinmiyor*, `0` *açıkça sıfır* demektir; negatif değer kaydedilmez ve değerlendirme başlamaz.

## İleri dönem sanal takip tabloları (spec 0005, Adım 7)

Belge deposundan ayrı, aynı veritabanında dört tablo (`forward_tracker.py`, ilk
kullanımda kendiliğinden oluşur):

| Tablo | Anahtar | İçerik |
|---|---|---|
| `forward_decisions` | (varlık, strateji sürümü, mum günü) | Günlük sanal karar, kaynak, karar zamanı, mum, varsayımlar, zamanında/sonradan, gerçek saat, piyasa koşulu |
| `forward_revisions` | (varlık, sürüm, mum günü, yeni mum) | Sağlayıcının sonradan değiştirdiği mum; karar değişmez |
| `forward_trades` | (varlık, sürüm, giriş günü) | Sanal kapanmış işlemler (gerçek işlemler `position_journal`'dadır) |
| `forward_runs` | çalışma zamanı | Koşucu çalışmaları ve notları |

Tekillik veritabanında zorlanır; aynı karar ikinci kez yazılırsa hata değil yok
sayma olur (`INSERT … ON CONFLICT DO NOTHING`, SQLite ve Postgres).

Koşucu (`python -m forward_runner`) GitHub Actions'ta
(`.github/workflows/forward-tracking.yml`) çalışır ve aynı Neon veritabanına
`ANALIZ_APP_DB_URL` **repository secret**'ı ile bağlanır; değer Streamlit
secrets'taki `db_url` ile aynıdır.
