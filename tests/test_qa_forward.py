"""Spec 0005 Rev 5 — QA bulguları: ileri takip (B1, B2, B3, B8, B9, B16)."""
from datetime import timedelta
from decimal import Decimal

import pytest

import forward_runner
import forward_tracker as ft
from conftest import make_ohlcv
from data_fetchers import process_data


def _frame(rows=None):
    raw = make_ohlcv()
    raw.index = raw.index.tz_localize("UTC")
    frame, _ = process_data(raw, "fixture")
    return frame if rows is None else frame.iloc[:rows]


def _run(frame, now, *, real_clock=True, symbols=("BTC-USD",), include_ml=False, **kwargs):
    return forward_runner.run(now, real_clock=real_clock, fetch=lambda s: (frame, "fixture"),
                              include_ml=include_ml, symbols=list(symbols),
                              clock=lambda: now + timedelta(minutes=1), **kwargs)


def _day(frame, offset=-1):
    return frame.index[offset].date()


def test_b1_v1_decisions_can_be_limited_to_the_last_bars_with_identical_results():
    """AC90 — `build_v1_decisions(only_last=N)` son N barın kararını tam hesapla birebir aynı üretir."""
    from technical_analysis import build_v1_decisions

    frame = _frame()
    full = build_v1_decisions(frame, include_ml=False)
    part = build_v1_decisions(frame, include_ml=False, only_last=4)
    assert len(part) == 4 and list(part) == list(full)[-4:]
    assert all(part[day] == full[day] for day in part)


def test_ac90_runner_computes_only_new_days_not_the_whole_history(monkeypatch):
    """AC90 — Koşucu yalnız yeni günlerin kararını hesaplar; kayıtlı günleri yeniden hesaplamaz."""
    import technical_analysis

    frame = _frame()
    seen = []
    real = technical_analysis.build_v1_decisions

    def spy(df, **kwargs):
        seen.append(kwargs.get("only_last"))
        return real(df, **{**kwargs, "include_ml": False})

    monkeypatch.setattr(technical_analysis, "build_v1_decisions", spy)
    first_day = _day(frame, -4)
    _run(frame.loc[:str(first_day)], ft.available_at("KRIPTO", first_day), include_ml=True)
    assert seen == [1]                                     # ilk çalışma: yalnız son gün
    _run(frame, ft.available_at("KRIPTO", _day(frame)), include_ml=True)
    assert seen == [1, 3]                                  # yalnız 3 yeni gün
    _run(frame, ft.available_at("KRIPTO", _day(frame)), include_ml=True)
    assert seen == [1, 3]                                  # yeni gün yok: hiç hesaplanmaz


def test_ac90_evaluated_at_is_taken_at_write_time_per_asset():
    """AC90 — Karar zamanı her varlık için yazım anında alınır; çalışma başlangıç saati değildir."""
    frame = _frame()
    now = ft.available_at("KRIPTO", _day(frame))
    ticks = iter(range(100))
    forward_runner.run(now, real_clock=True, fetch=lambda s: (frame, "fixture"), include_ml=False,
                       symbols=["BTC-USD", "ETH-USD"],
                       clock=lambda: now + timedelta(minutes=next(ticks)))
    times = [ft.get_decision(a, v, _day(frame)).evaluated_at
             for a in ("BTC-USD", "ETH-USD") for v in ft.versions(a)]
    assert len(set(times)) == 2


def test_ac91_runner_records_a_provider_revision_and_keeps_the_decision():
    """AC91 — Koşucu yeniden çalışınca sağlayıcının kayıtlı mumu değiştirdiğini görür; karar değişmez, revizyon ayrı yazılır."""
    frame = _frame()
    day = _day(frame)
    now = ft.available_at("KRIPTO", day)
    _run(frame, now)
    version = ft.versions("BTC-USD")[0]
    stored = ft.get_decision("BTC-USD", version, day)
    revised = frame.copy()
    revised.iloc[-1, revised.columns.get_loc("Close")] = revised["Close"].iloc[-1] * 1.10
    _run(revised, now + timedelta(hours=3))
    assert ft.get_decision("BTC-USD", version, day).decision == stored.decision
    rows = ft.revisions("BTC-USD", version)
    assert len(rows) == 1 and rows[0]["candle_day"] == day.isoformat()
    assert Decimal(rows[0]["new_candle"]["close"]) != Decimal(stored.candle["close"])
    _run(revised, now + timedelta(hours=4))               # aynı revizyon ikinci kez eklenmez
    assert len(ft.revisions("BTC-USD", version)) == 1


def _trade_decisions(real_clock=True, late_first=False):
    base = _frame().index[-6].date()
    for offset in range(6):
        day = base + timedelta(days=offset)
        evaluated = ft.available_at("KRIPTO", day) + timedelta(
            hours=30 if (late_first and offset == 0) else 0, minutes=10)
        ft.record(ft.ForwardDecision(
            asset="BTC-USD", strategy_version="V1", candle_day=day, decision="BEKLE", source="t",
            evaluated_at=evaluated, candle={"close": "1"}, on_time=ft.is_on_time("KRIPTO", day, evaluated),
            real_clock=real_clock, regime="YUKSELEN", regime_version="REJIM-1"))
    return base


def test_ac92_trades_of_accelerated_observations_are_not_counted():
    """AC92 — Hızlandırılmış saatle üretilmiş gözlemin sanal işlemi kapanmış işlem sayacına girmez."""
    base = _trade_decisions(real_clock=False)
    ft.save_trades("BTC-USD", "V1", [{"entry_day": base + timedelta(days=1),
                                      "exit_day": base + timedelta(days=3), "pnl": Decimal("5")}])
    assert ft.tracked_days("BTC-USD", "V1") == 0 and ft.closed_trades("BTC-USD", "V1") == 0
    assert "kapanmış işlem 0/30" in " ".join(ft.assess("BTC-USD", "V1").missing)


def test_ac92_trade_spanning_a_late_created_decision_is_not_counted():
    """AC92 — Geç oluşturulan bir karara dayanan sanal işlem zamanında sayacına girmez; tamamen zamanında olan girer."""
    base = _trade_decisions(late_first=True)
    ft.save_trades("BTC-USD", "V1", [
        {"entry_day": base + timedelta(days=1), "exit_day": base + timedelta(days=2), "pnl": Decimal("5")},
        {"entry_day": base + timedelta(days=3), "exit_day": base + timedelta(days=5), "pnl": Decimal("7")}])
    assert ft.closed_trades("BTC-USD", "V1") == 1


def test_ac92_accelerated_run_does_not_block_the_real_record_of_the_same_day():
    """AC92 — Hızlandırılmış koşu aynı günün sonradan gerçek saatle kaydedilmesini engellemez (ayrı tatbikat sürümü)."""
    frame = _frame()
    day = _day(frame)
    now = ft.available_at("KRIPTO", day)
    _run(frame, now, real_clock=False)
    assert all(v.endswith("/TATBIKAT") for v in ft.versions("BTC-USD"))
    report = _run(frame, now + timedelta(minutes=30), real_clock=True)
    assert report.recorded
    real = [v for v in ft.versions("BTC-USD") if not v.endswith("/TATBIKAT")]
    assert len(real) == 1 and ft.get_decision("BTC-USD", real[0], day).real_clock is True
    assert ft.tracked_days("BTC-USD", real[0]) == 1


def test_ac103_version_label_carries_rule_and_assumption_fingerprint(store):
    """AC103 — Sürüm etiketi karar kuralı ve işlem varsayımlarının parmak izini taşır; varsayım değişince yeni sürüm olur."""
    import trade_settings

    frame = _frame()
    day = _day(frame)
    now = ft.available_at("KRIPTO", day)
    trade_settings.save_raw("BTC-USD", {"capital": "10000", "quantity_step": "0.001", "commission_pct": "0.1"})
    _run(frame, now, symbols=("BTC-USD",))
    first = ft.versions("BTC-USD")
    trade_settings.save_raw("BTC-USD", {"capital": "10000", "quantity_step": "0.001", "commission_pct": "0.2"})
    _run(frame, now + timedelta(minutes=5), symbols=("BTC-USD",))
    both = ft.versions("BTC-USD")
    assert len(first) == 1 and len(both) == 2 and first[0] in both
    assert ft.tracked_days("BTC-USD", both[0]) == 1 and ft.tracked_days("BTC-USD", both[1]) == 1
    latest = ft.get_decision("BTC-USD", [v for v in both if v not in first][0], day)
    assert latest.assumptions["komisyon"] == "0.2" and latest.assumptions["makas"] == "bilinmiyor"


def test_ac99_missing_days_failed_run_and_missing_step_are_visible():
    """AC99 — Eksik görünen günler, son başarısız çalışma notu ve miktar adımı eksikliği ekranda görünür."""
    import forward_ui
    from datetime import datetime, timezone

    frame = _frame()
    for offset in (-8, -7, -4):                              # -6, -5 günleri kayıp
        day = _day(frame, offset)
        ft.record(ft.ForwardDecision(
            asset="BTC-USD", strategy_version="V1", candle_day=day, decision="BEKLE", source="t",
            evaluated_at=ft.available_at("KRIPTO", day) + timedelta(minutes=5), candle={"close": "1"}))
    when = datetime(2026, 10, 4, 1, 0, tzinfo=timezone.utc)
    ft.record_run(when, False, "BTC-USD: veri alınamadı (Binance); ETH-USD: miktar adımı/varsayımlar eksik; sanal işlem hesaplanmadı.")
    text = " | ".join(forward_ui.build_forward_view(when + timedelta(hours=1)).lines)
    assert "eksik görünen gün: 2" in text
    assert "Son çalışma başarısız" in text and "veri alınamadı" in text
    assert "miktar adımı/varsayımlar eksik" in text


def test_ac100_unreadable_tables_raise_storage_error_not_raw_database_error(store, monkeypatch):
    """AC100 — İleri takip tabloları okunamadığında ham veritabanı hatası değil `StorageAccessError` yükselir ve panel çökmez."""
    import sqlalchemy

    import forward_ui
    import storage
    from datetime import datetime, timezone

    ft.ensure_tables()
    monkeypatch.setattr(storage, "get_engine",
                        lambda: sqlalchemy.create_engine("sqlite:////yok/boyle/dizin/x.db", future=True))
    with pytest.raises(storage.StorageAccessError) as raised:
        ft.decisions("BTC-USD", "V1")
    assert "SQL" not in str(raised.value) and "sqlite" not in str(raised.value).lower()
    with pytest.raises(storage.StorageAccessError):
        forward_ui.build_forward_view(datetime.now(timezone.utc))
