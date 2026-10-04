"""Spec 0005 Adım 7 / S7 — süre bütçeleri (Q06b, AC45, AC46, AC86).

Referans ölçüm ortamı: sabit veri fixture'ı (ağ yok), ilk (soğuk) çalışma ayrı
raporlanır, 10 tekrarın **en kötüsü** bütçe altında olmalıdır. Ortam bilgisi
(Python, CPU, işlemci sayısı) çıktıya yazılır. Yavaş olduğu için `-m perf` ile
ayrı çalışır: `python -m pytest tests/ -m perf -s`.
"""
import os
import platform
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from conftest import make_ohlcv

pytestmark = pytest.mark.perf
REPEATS = 10


def _environment():
    return (f"Python {platform.python_version()} · {platform.machine()} · "
            f"{platform.processor() or 'işlemci adı yok'} · {os.cpu_count()} çekirdek")


def _measure(action):
    cold_started = time.perf_counter()
    action()
    cold = time.perf_counter() - cold_started
    warm = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        action()
        warm.append(time.perf_counter() - started)
    return cold, warm


def test_ac45_report_on_10000_records_within_2_seconds():
    """AC45 — Hazır 10.000 kayıtta rapor/filtre 10 tekrarın her birinde en fazla 2 saniye."""
    from performance_report import EquityPoint, ReportFilters, ReportTrade, build_report

    start = date(2020, 1, 1)
    symbols = [("ETH-USD", "KRIPTO", "USD"), ("THYAO.IS", "BIST", "TRY"), ("AAPL", "ABD", "USD")]
    trades = [ReportTrade(*symbols[i % 3], Decimal(i % 97) - Decimal("40"),
                          start + timedelta(days=i % 2000), "V1", i % 11 != 0)
              for i in range(10_000)]
    equity = {s: [EquityPoint(start + timedelta(days=d), Decimal(10000 + d)) for d in range(2000)]
              for s, _, _ in symbols}
    filters = ReportFilters(market="KRIPTO", start=date(2021, 1, 1), end=date(2024, 12, 31))

    cold, warm = _measure(lambda: (build_report(equity, trades, ReportFilters()),
                                   build_report(equity, trades, filters)))
    print(f"\nAC45 ortam: {_environment()} · soğuk {cold:.3f} sn · sıcak en kötü {max(warm):.3f} sn")
    assert max(warm) <= 2.0
    assert cold <= 2.0


def test_ac46_20_assets_1000_candles_3_candidates_within_120_seconds(store):
    """AC46 — 20 varlık × 1.000 günlük mum × 3 aday değerlendirmesi 10 tekrarın her birinde ≤ 120 sn."""
    import candidate_ui
    from data_fetchers import process_data

    frames = []
    for seed in range(20):
        raw = make_ohlcv(seed=seed, segments=[(250, 0.003), (250, -0.003), (250, 0.0), (250, 0.004)])
        raw.index = raw.index.tz_localize("UTC")
        frame, _ = process_data(raw, "fixture")
        index = frame.index
        v1 = {index[i]: ("AL" if i % 40 == 0 else "SAT" if i % 40 == 20 else "BEKLE")
              for i in range(len(index))}
        frames.append((frame, v1))
    now = datetime(2026, 10, 3, tzinfo=timezone.utc)

    def evaluate_all():
        # 3 aday: V1 referansı + kırılım + geri çekilme
        for frame, v1 in frames:
            outcome = candidate_ui.evaluate_candidates(
                "BTC-USD", frame, v1, notional=Decimal("1000"), capital=Decimal("10000"),
                quantity_step=Decimal("0.001"), costs=None, now=now)
            assert outcome.ok

    cold, warm = _measure(evaluate_all)
    print(f"\nAC46 ortam: {_environment()} · soğuk {cold:.1f} sn · sıcak en kötü {max(warm):.1f} sn")
    assert max(warm) <= 120.0
    assert cold <= 120.0


def test_ac86_provider_wait_is_reported_apart_from_compute(store):
    """AC86 — Ölçüm raporunda sağlayıcı bekleme süresi hesaplama süresinden ayrı gösterilir."""
    import forward_runner
    from data_fetchers import process_data

    raw = make_ohlcv()
    raw.index = raw.index.tz_localize("UTC")
    frame, _ = process_data(raw, "fixture")

    def slow_provider(symbol):
        time.sleep(0.5)  # sağlayıcı bekleyişi taklidi
        return frame, "fixture"

    last = frame.index[-1].date()
    import forward_tracker as ft
    started = time.perf_counter()
    report = forward_runner.run(ft.available_at("KRIPTO", last) + timedelta(minutes=5),
                                real_clock=False, fetch=slow_provider, include_ml=False,
                                symbols=["BTC-USD"])
    wall = time.perf_counter() - started
    print(f"\nAC86 sağlayıcı {report.provider_seconds:.2f} sn · hesaplama {report.compute_seconds:.2f} sn")
    assert report.recorded
    assert report.provider_seconds >= 0.5                     # bekleyiş sağlayıcıya yazıldı
    assert report.compute_seconds <= wall - 0.5 + 0.05        # hesaplama süresine karışmadı


def test_ac90_runner_with_ml_on_stays_within_the_workflow_budget(store):
    """AC90 — ML açıkken koşucunun bir varlık için yeni günü işlemesi, 20 varlığın 60 dakikalık iş sınırına sığar."""
    import forward_runner
    import forward_tracker as ft
    from data_fetchers import process_data

    raw = make_ohlcv()
    raw.index = raw.index.tz_localize("UTC")
    frame, _ = process_data(raw, "fixture")
    now = ft.available_at("KRIPTO", frame.index[-1].date()) + timedelta(minutes=5)

    started = time.perf_counter()
    report = forward_runner.run(now, real_clock=False, fetch=lambda symbol: (frame, "fixture"),
                                include_ml=True, symbols=["BTC-USD"])
    elapsed = time.perf_counter() - started
    print(f"\nAC90 ML açık, 1 varlık, ilk koşu: {elapsed:.1f} sn · 20 varlık tahmini {elapsed * 20 / 60:.1f} dk "
          f"· {_environment()}")
    assert report.recorded
    assert elapsed * 20 <= 3600          # 20 varlık × tek varlık süresi iş zaman aşımının (60 dk) altında
