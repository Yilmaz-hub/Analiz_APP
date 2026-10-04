# Spec 0005 Adım 7 — süre bütçesi ölçümü (S7)

- Tarih: 2026-10-03
- Komut: `python -m pytest tests/test_performance_budgets.py -m perf -s`
- Ortam: Python 3.13.14 · x86_64 · 4 çekirdek (bulut geliştirme konteyneri)
- Veri: sabit fixture (`tests/conftest.make_ohlcv`, sabit tohumlar); ağ yok. İlk (soğuk)
  çalışma ayrı; ardından 10 tekrar, **en kötüsü** raporlanır.

| Kriter | Bütçe | Soğuk | Sıcak en kötü (10 tekrar) | Sonuç |
|---|---|---|---|---|
| AC45 — 10.000 kayıt rapor + filtre | 2 sn | 0,032 sn | 0,046 sn | ✅ |
| AC46 — 20 varlık × 1.000 mum × 3 aday | 120 sn | 19,6 sn | 18,7 sn | ✅ |
| AC86 — sağlayıcı bekleyişi ayrı | — | sağlayıcı 0,50 sn · hesaplama 2,80 sn | — | ✅ ayrı raporlandı |

AC35 (2 saat) koşucu içi süredir; GitHub Actions zamanlama gecikmesi ölçüme dahil değildir,
geciken gün "sonradan oluşturuldu" olarak işaretlenir (Q06a/Q06b).

## Ek (2026-10-04) — QA bulgusu B1 sonrası koşucu ölçümü

QA denetimi, ML açıkken (workflow `--no-ml` vermiyor) koşucunun tek varlık için tüm geçmişi yeniden
hesapladığını ölçtü: **393 sn**; 20 varlık ≈ 130 dk, 60 dk iş sınırını aşar. Düzeltme: koşucu yalnız
yeni (ve kaçan) günlerin kararını hesaplar (`build_v1_decisions(only_last=N)`).

| Ölçüm | Önce (QA) | Sonra |
|---|---|---|
| ML açık, 1 varlık, tek yeni gün | 393 sn | **0,7 sn** (`tests/test_performance_budgets.py::test_ac90…`, `-m perf`) |
| 20 varlık tahmini | ≈ 130 dk | ≈ 0,2 dk (iş sınırı 60 dk) |

Ortam: Python 3.13.14 · x86_64 · 4 çekirdek. Yalnız yeni günler hesaplandığı için kaçan 10 güne kadar
tamamlama da bu sınırın çok altındadır (her ek gün ≈ 0,3 sn).
