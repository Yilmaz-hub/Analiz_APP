# Konvansiyonlar (conventions)

> Adlandırma, modül yapısı, hata yönetimi, para. Yığın: Python 3, Streamlit (`app.py`),
> pytest (`tests/`). Alan terimleri her spec'in **Context → Terimler** bölümünde tanımlanır.

## Para = decimal (mutlak kural)

- Tüm para ve yüzde alanları **`decimal.Decimal`**. `float` **yasak**.
- Yuvarlama tek noktada ve açık yön/hassasiyetle yapılır (`quantize` + `ROUND_FLOOR` /
  `ROUND_CEILING`; örnek: `trade_execution.py`).
- `Decimal`, `float`'tan değil ondalık gösteriminden (`Decimal(str(x))`) üretilir.
- Kalıcı kayıtlarda para/yüzde, yazıldığı gösterimle korunur (bkz. `storage.md`).

## V1 analitik ara hesap istisnası

İşlem tutarlılığı feature'ı için onaylı istisna: mevcut gösterge/ML kütüphanelerinin yalnız analitik ara hesaplarında kayan nokta kullanılabilir. Analitik fiyat/ATR çıktısı finansal hesaba geçerken ondalık gösterimi üzerinden decimal'e dönüştürülür; fiyat adımı yuvarlaması bundan sonra uygulanır. Dönüşüm hassasiyet kazandırmış sayılmaz. İşlem fiyatı, stop, miktar, tutar, komisyon, bakiye ve getiri/yüzde hesapları decimal kalır; finansal defter bu istisnaya dahil değildir. Sunum yuvarlaması hesap girdisi olamaz.

## Adlandırma (PEP 8)

- Modül, fonksiyon, değişken → `snake_case`; sınıf → `PascalCase`; sabit → `UPPER_SNAKE_CASE`;
  modül içi yardımcı → `_baslangic_alt_cizgi`.
- Kod adları İngilizce yazılır (`decide_action`, `PositionState`); kullanıcıya görünen metinler
  Türkçedir.
- Boolean adları soru gibi: `is_mobile_mode`, `is_chart_renderable`, `storage_ok`.
- Sabit kümeler `Enum` ile (`Signal`, `Action`, `PositionState` — `trading_contracts.py`);
  veri taşıyıcıları `dataclass` ile.

## Modül Yapısı

Mimari kaynağı koddur (bkz. `AGENTS.md` Altın Kural 3). Kök dizindeki modüller düz bir
yapıdadır; sorumluluklar şöyle ayrılır:

| Katman | Modüller |
|---|---|
| Arayüz (Streamlit) | `app.py`, `ui_components.py`, `trading_ui.py`, `theme.py` |
| İşlem kuralları (saf, `Decimal`) | `trading_contracts.py`, `trade_decisions.py`, `trade_execution.py`, `trading_service.py` |
| Sinyal / analiz | `signal_engine.py`, `technical_analysis.py`, `indicator_calculations.py`, `advanced_analysis.py`, `ml_models.py`, `weight_profiles.py` |
| Veri | `data_fetchers.py`, `market_validation.py` |
| Kayıtlar | `storage.py`, `assets.py`, `portfolio.py`, `positions.py`, `position_journal.py`, `paper_trading.py`, `prediction_tracker.py` |
| Ortak | `config.py`, `exceptions.py`, `logger.py` |

- **İş kuralı arayüze gömülmez.** Yeni iş kuralı `app.py` / `ui_components.py` içine değil,
  ilgili modüle yazılır; arayüz yalnız çağırır ve gösterir. İşlem kuralları modülleri
  (`trading_contracts`, `trade_decisions`, `trade_execution`, `trading_service`) `streamlit`
  içe aktarmaz; böylece doğrudan test edilir.
- Ayarlar ve eşikler `config.py`'deki yapılandırma sınıflarında tutulur
  (`SignalConfig`, `RiskConfig`, `BacktestConfig`, `TradingV1Config` …).
- Varlık listesi ve portföy yalnız `storage.py` üzerinden okunur/yazılır (bkz. `storage.md`).
  Sanal işlem (`paper_trading.json`), tahmin günlüğü, pozisyon günlüğü ve ağırlık profilleri
  henüz kendi JSON dosyalarındadır; yazımları geçici dosya + yeniden adlandırma ile atomiktir.
  Bunların depoya taşınması ayrı spec konusudur.

## Hata Yönetimi

- Uygulamaya özgü hatalar `exceptions.py`'deki `ProTraderError`'dan türetilir
  (`DataFetchError`, `PriceDataError`, `StorageAccessError` …). Çıplak `except:` kullanılmaz.
- Beklenen durumlar (eksik veri, geçersiz pozisyon) sonuç nesnesiyle döner
  (ör. `Validation`, `ProviderResult`); akış exception'la kontrol edilmez.
- **Kullanıcıya teknik hata metni / traceback gösterilmez.** Arayüz anlaşılır Türkçe mesaj verir
  (`st.error` / `st.warning`); teknik ayrıntı log'a yazılır (bkz. `frontend.md`).
- Kayıt okunamıyorsa **varsayılana sessizce düşülmez** (bkz. `storage.md`).
- Loglama `logger.setup_logger()` ile yapılır; log'a sır (DB URL, parola) yazılmaz.

## Genel

- Bir kural yalnızca burada; başka dosyalar link verir (tek gerçek kaynağı).
- Ölü kod, yorum satırına alınmış kod bırakılmaz; geçmiş `git`tedir.
