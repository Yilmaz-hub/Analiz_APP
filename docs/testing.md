# Test Yaklaşımı

> Bu dosya, alakasız bir projeye ait .NET/xUnit sürümü depodan kaldırıldıktan sonra bu projenin
> gerçek yığınına göre yeniden yazılmıştır (karar: `specs/0004` P3 + Takım Yöneticisi onayı,
> 2026-09-10). Teknolojiden bağımsız kurallar korunmuştur; teknolojiye özgü bölüm gerçek koda göre
> yazılmıştır. Yığın: Python 3, Streamlit arayüz (`app.py`), pytest (`tests/`).

## Kural

- **Her kabul kriteri bir test.** Spec'teki her Acceptance Criteria satırı en az bir testle
  karşılanır (bkz. `AGENTS.md` Altın Kural 6).
- Bir kriter tek başına test edilemiyorsa kriter yeniden yazılır — test kalabalıklaştırılmaz.

## Python (pytest)

- Çatı: **pytest** (`requirements-dev.txt`). Testler `tests/` altında, dosya adı `test_<modül>.py`.
- Adlandırma deseni: **`test_<işlev>_<durum>_<beklenen_sonuç>`**
  - ör. `test_view_mode_control_is_required`
  - ör. `test_short_or_missing_data_returns_safe_defaults`
- Para/yüzde iddiaları **`decimal`** ile yapılır; kayan nokta karşılaştırması yapılmaz
  (bkz. `conventions.md` — para=decimal ve V1 analitik ara hesap istisnası).
- Testler bağımsız ve tekrarlanabilir; dış servis (fiyat sağlayıcı, ağ) fake/mock ile izole edilir —
  ortak kurgular `tests/conftest.py`'de tutulur.
- Zaman ve rastgelelik sabitlenir; testler makinenin saatine veya canlı veriye bağlı olmaz.

## Arayüz (Streamlit)

- **Her mini-spec kriterine görsel kanıt**: ilgili ekranın **ekran görüntüsü** PR'a eklenir.
- Yükleme / boş / hata durumları da kanıtlanır.
- **Kritik akışa smoke test**: en az bir uçtan uca "mutlu yol" otomasyonu.

## Bu depodaki uygulama (Python / Streamlit)

> Karar: spec 0004, **P3**. Yukarıdaki teknolojiye özgü bölümler (xUnit,
> `Metot_Durum_BeklenenSonuc`, .NET test projesi ayrımı) ilgisiz bir projeye
> aittir ve bu depo için **bağlayıcı değildir**.

- Çatı: **pytest** (`requirements-dev.txt`), testler `tests/` altında.
- Sürekli tümleştirme: `.github/workflows/tests.yml` → `python -m pytest tests/ -v`, ardından süre
  bütçesi testleri ayrı adımda: `python -m pytest tests/ -m perf -v -s` (`pytest.ini`: `perf`
  işaretli testler normal koşudan çıkarılır; spec 0005 S7). Ölçüm ortamı (Python sürümü, işlemci,
  çekirdek) test çıktısına yazılır; kanıt `docs/evidence/` altında tutulur.
- Ekransız ileri takip koşucusu `.github/workflows/forward-tracking.yml` ile zamanlanır
  (`python -m forward_runner`); `ANALIZ_APP_DB_URL` repository secret'ı gerekir (`storage.md`).
- Arayüz davranışı gerçek `app.py` üzerinde **`streamlit.testing.v1.AppTest`**
  ile sürülür; ağ kaynakları `monkeypatch` ile taklit edilir.
- Kayıt deposu testlerde izoledir: `tests/conftest.py` içindeki `store`
  fixture'ı `autouse`'dur, hiçbir test gerçek kullanıcı kayıtlarına dokunamaz.

## Genel

- Yeşil pipeline olmadan PR merge edilmez (bkz. `git.md`).
- Bir hata bulunduğunda önce hatayı gösteren test yazılır, sonra düzeltme yapılır (kaçan hata
  SCORECARD'a işlenir — `../specs/TEMPLATE.md`).
- Kararsız (flaky) test skip/retry ile susturulmaz; deterministikleştirilir — kanıt: art arda
  5 yeşil (bkz. `ap.md` AP-03).
