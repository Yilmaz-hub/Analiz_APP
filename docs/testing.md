# Test Yaklaşımı

> Yığın: Python 3, Streamlit arayüz (`app.py`), pytest (`tests/`).
> Çalıştırma: `pip install -r requirements-dev.txt` → `python -m pytest tests/ -v`.

## Kural

- **Her kabul kriteri bir test.** Spec'teki her Acceptance Criteria satırı en az bir testle
  karşılanır (bkz. `AGENTS.md` Altın Kural 6).
- Bir kriter tek başına test edilemiyorsa kriter yeniden yazılır — test kalabalıklaştırılmaz.

## Python (pytest)

- Çatı: **pytest** (`requirements-dev.txt`). Testler `tests/` altında, dosya adı `test_<modül>.py`.
- Adlandırma deseni: **`test_<işlev>_<durum>_<beklenen_sonuç>`**
  - ör. `test_view_mode_control_is_required`
  - ör. `test_short_or_missing_data_returns_safe_defaults`
- Para/yüzde iddiaları **`Decimal`** ile yapılır; kayan nokta karşılaştırması yapılmaz
  (bkz. `conventions.md` — para=decimal ve V1 analitik ara hesap istisnası).
- Testler bağımsız ve tekrarlanabilir; dış servis (fiyat sağlayıcı, ağ) `monkeypatch` /
  fake ile izole edilir — ortak kurgular `tests/conftest.py`'de tutulur.
- Zaman ve rastgelelik sabitlenir; testler makinenin saatine veya canlı veriye bağlı olmaz
  (sentetik OHLCV: `conftest.make_ohlcv(seed=…)`).
- **Kayıt deposu testlerde izoledir:** `tests/conftest.py` içindeki `store` fixture'ı
  `autouse`'dur; hiçbir test gerçek kullanıcı kayıtlarına dokunamaz.

## Arayüz (Streamlit)

- Arayüz davranışı gerçek `app.py` üzerinde **`streamlit.testing.v1.AppTest`** ile sürülür
  (ör. `tests/test_app_view_mode.py`, `tests/test_persistence_app.py`).
- **Her mini-spec kriterine görsel kanıt**: ilgili ekranın **ekran görüntüsü** PR'a eklenir.
- Yükleme / boş / hata durumları da kanıtlanır.
- **Kritik akışa smoke test**: en az bir uçtan uca "mutlu yol" otomasyonu
  (ör. `tests/test_trading_smoke.py`).

## Performans

- Süre bütçeleri `config.TradingV1Config` içinde tutulur ve `tests/test_trading_performance.py`
  ile doğrulanır; ölçüm kanıtları `docs/evidence/<spec-no>/` altına yazılır.

## Genel

- Yeşil pipeline olmadan PR merge edilmez: `.github/workflows/tests.yml` (bkz. `git.md`).
- Bir hata bulunduğunda önce hatayı gösteren test yazılır, sonra düzeltme yapılır (kaçan hata
  SCORECARD'a işlenir — `../specs/TEMPLATE.md`).
- Kararsız (flaky) test skip/retry ile susturulmaz; deterministikleştirilir — kanıt: art arda
  5 yeşil (bkz. `ap.md` AP-03).
