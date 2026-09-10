# V1 performans kanıtı — 2026-09-10

- Ortam: Windows 10 10.0.19045, Python 3.14.2. İşlemci/RAM bilgisi kısıtlı çalışma ortamında okunamadı.
- Veri: deterministik 500 ve 1.000 günlük sentetik OHLCV; gerçek piyasa takvimini temsil etmez.
- Ağ: kullanılmadı. Sağlayıcı hatası ve zaman aşımı ayrı davranış testidir.
- ML: 500 mumluk ilk sinyal hesabında açık.

| Ölçüm | Sonuç |
|---|---:|
| Beş bağımsız `data_fetchers` ilk yüklemesi | 1,6973 / 1,6761 / 1,7519 / 1,7952 / 1,8264 sn |
| İlk 500 mumluk sinyal hesabı | 0,4385 sn |
| 20 ısınmış sinyal tekrarı medyanı | 0,001017 sn |
| 20 ısınmış sinyal tekrarı p95 | 0,001910 sn |
| 20 ısınmış sinyal tekrarı en uzun | 0,002680 sn |
| 1.000 mumluk gerçekleşme hesabı | 0,0996 sn |

Kabul bütçeleri `TradingV1Config` içinde tutulur. `tests/test_trading_performance.py`, panel hesabını, beş bağımsız ilk yüklemeyi ve 500 mumluk gösterge işlemesinin `pandas_ta` yüklemeden tamamlanmasını doğrular.
