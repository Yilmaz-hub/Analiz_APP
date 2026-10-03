"""Spec 0005 Adım 7 / S6 — ileri dönem sanal takip kaydı (R15–R19, Q07, Q08)."""
import threading
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import forward_tracker as ft
from regime_classifier import DUSEN, YATAY, YUKSELEN

D0 = date(2026, 6, 1)
CANDLE = {"open": "100", "high": "101", "low": "99", "close": "100.5"}


def _decision(day=D0, version="V1", *, asset="BTC-USD", decision="BEKLE", evaluated_at=None,
              candle=None, real_clock=True, regime=YUKSELEN):
    evaluated_at = evaluated_at or ft.available_at("KRIPTO", day) + timedelta(minutes=30)
    return ft.ForwardDecision(
        asset=asset, strategy_version=version, candle_day=day, decision=decision,
        source="Binance", evaluated_at=evaluated_at, candle=candle or CANDLE,
        assumptions={"fill": "ertesi açılış"},
        on_time=ft.is_on_time("KRIPTO", day, evaluated_at), real_clock=real_clock,
        regime=regime, regime_version="REJIM-1")


def test_ac36_decision_shows_source_times_and_version():
    """AC36 — Açılan ileri dönem kararı kaynak, karar zamanı, mum zamanı ve strateji sürümünü gösterir."""
    ft.record(_decision())
    stored = ft.get_decision("BTC-USD", "V1", D0)
    assert (stored.source, stored.candle_day, stored.strategy_version) == ("Binance", D0, "V1")
    assert stored.evaluated_at == ft.available_at("KRIPTO", D0) + timedelta(minutes=30)
    assert stored.assumptions == {"fill": "ertesi açılış"}


def test_ac37_reprocessing_same_decision_does_not_add_record():
    """AC37 — Aynı varlık, sürüm ve gün yeniden işlenince sanal kayıt sayısı artmaz."""
    assert ft.record(_decision(decision="AL")) is True
    assert ft.record(_decision(decision="AL")) is False
    assert ft.decision_count("BTC-USD", "V1") == 1


def test_ac70_concurrent_runs_create_single_record():
    """AC70 — Aynı varlık/sürüm/mum için eşzamanlı iki koşum tek kayıt oluşturur, hata vermez."""
    ft.ensure_tables()
    barrier = threading.Barrier(2)
    results, errors = [], []

    def run():
        try:
            barrier.wait()
            results.append(ft.record(_decision(decision="AL")))
        except Exception as exc:  # pragma: no cover - testin yakaladığı hata
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(results) == [False, True]
    assert ft.decision_count("BTC-USD", "V1") == 1


def test_ac71_candle_revision_keeps_decision_and_is_recorded_separately():
    """AC71 — Dayanak mum sonradan revize edilirse karar değişmez, revizyon ayrı kaydedilir."""
    ft.record(_decision(decision="AL"))
    revised = dict(CANDLE, close="97")
    ft.record(_decision(decision="BEKLE", candle=revised))
    assert ft.get_decision("BTC-USD", "V1", D0).decision == "AL"
    revisions = ft.revisions("BTC-USD", "V1")
    assert [(r["candle_day"], r["new_candle"]["close"]) for r in revisions] == [(D0.isoformat(), "97")]


def test_ac38_backfilled_day_is_marked_late_and_not_counted():
    """AC38 — Sonradan tamamlanan gün "sonradan oluşturuldu" görünür, zamanında sayıya katılmaz."""
    late = ft.available_at("KRIPTO", D0) + timedelta(days=2)
    ft.record(_decision(evaluated_at=late))
    stored = ft.get_decision("BTC-USD", "V1", D0)
    assert stored.on_time is False and ft.status_text(stored) == "sonradan oluşturuldu"
    assert ft.tracked_days("BTC-USD", "V1") == 0


def test_ac35_on_time_window_is_two_hours_after_data():
    """AC35 — Veri geldikten sonra en çok 2 saat içindeki kayıt zamanında sayılır (sınır dahil)."""
    ready = ft.available_at("KRIPTO", D0)
    assert ready == datetime(2026, 6, 2, 0, 0, tzinfo=timezone.utc)
    assert ft.is_on_time("KRIPTO", D0, ready + timedelta(hours=2))
    assert not ft.is_on_time("KRIPTO", D0, ready + timedelta(hours=2, seconds=1))


def test_ac63_missing_day_is_not_counted():
    """AC63 — İzlemenin çalışmadığı gün izlenen gün sayacına katılmaz."""
    ft.record(_decision(D0))
    ft.record(_decision(D0 + timedelta(days=2)))  # arada 1 gün kayıt yok
    assert ft.tracked_days("BTC-USD", "V1") == 2


def test_ac87_accelerated_clock_observation_is_not_counted():
    """AC87 — Saat/tarih ileri alınarak üretilen gözlem yeterlilik sayacına katılmaz."""
    ft.record(_decision(real_clock=False))
    assert ft.tracked_days("BTC-USD", "V1") == 0


def _sufficiency(days, trades=30, regimes=(YUKSELEN, DUSEN, YATAY)):
    return ft.sufficiency(days, trades, set(regimes))


def test_ac39_89_tracked_days_is_not_sufficient():
    """AC39 — Diğer koşullar sağlansa da 89 izlenen gün yeterli sayılmaz."""
    verdict = _sufficiency(89)
    assert not verdict.sufficient and "gün" in " ".join(verdict.missing)


def test_ac40_90_tracked_days_meets_duration():
    """AC40 — İşlem ve koşul kapsaması tamamken 90 izlenen gün süre koşulunu sağlar."""
    assert _sufficiency(90).sufficient


def test_ac41_missing_sideways_regime_is_not_sufficient():
    """AC41 — Süre ve işlem tamam ama yatay piyasa gözlemi yoksa yeterli sayılmaz."""
    verdict = _sufficiency(120, regimes=(YUKSELEN, DUSEN))
    assert not verdict.sufficient and any("Yatay" in item for item in verdict.missing)


def test_ac42_new_version_does_not_inherit_observations():
    """AC42 — Yeni strateji sürümünün ileri gözlem sayısı önceki sürümden devralınmaz."""
    for offset in range(3):
        ft.record(_decision(D0 + timedelta(days=offset)))
    ft.save_trades("BTC-USD", "V1", [{"entry_day": D0, "exit_day": D0, "pnl": Decimal("5")}])
    assert ft.tracked_days("BTC-USD", "V1") == 3
    assert ft.tracked_days("BTC-USD", "ADAY-1") == 0
    assert ft.closed_trades("BTC-USD", "ADAY-1") == 0


def test_ac43_virtual_buy_does_not_count_as_real_trade(store):
    """AC43 — Uygulanmayan AL sinyalinin sanal alışı gerçek teyitli işlem sayısını artırmaz."""
    from position_journal import PositionJournal

    journal = PositionJournal()
    before = len(journal.trades)
    ft.record(_decision(decision="AL"))
    ft.save_trades("BTC-USD", "V1", [{"entry_day": D0, "exit_day": D0 + timedelta(days=3),
                                      "pnl": Decimal("12")}])
    assert len(PositionJournal().trades) == before
    assert ft.closed_trades("BTC-USD", "V1") == 1


def test_ac44_late_real_trade_keeps_execution_and_record_times(store):
    """AC44 — Dünkü gerçek işlem bugün kaydedilince işlem zamanı dün, kayıt zamanı bugün kalır."""
    from position_journal import PositionJournal

    yesterday = datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc)
    today = datetime(2026, 10, 3, 9, 0, tzinfo=timezone.utc)
    journal = PositionJournal()
    journal.confirm_trade("ev-1", "BUY", Decimal("1"), Decimal("100"), yesterday, today, symbol="BTC-USD")
    trade = PositionJournal().trades[-1]
    assert (trade.executed_at, trade.recorded_at) == (yesterday, today)


def test_ac76_late_real_trade_flags_closed_period_and_keeps_counter():
    """AC76 — Kapanmış döneme geç kayıt "geç kayıtla güncellendi" işaretlenir, sayaç değişmez."""
    ft.record(_decision())
    before = ft.tracked_days("BTC-USD", "V1")
    trades = [{"executed_at": datetime(2026, 6, 10, tzinfo=timezone.utc),
               "recorded_at": datetime(2026, 7, 5, tzinfo=timezone.utc)}]
    assert ft.period_flag(trades, date(2026, 6, 1), date(2026, 6, 30)) == "geç kayıtla güncellendi"
    assert ft.period_flag(trades, date(2026, 7, 1), date(2026, 7, 31)) == ""
    assert ft.tracked_days("BTC-USD", "V1") == before


def test_ac77_open_real_position_keeps_entry_version():
    """AC77 — Aktif sürüm değişince açık gerçek pozisyon giriş anındaki sürümle yönetilir."""
    position = {"Coin": "BTC", "Status": "ACTIVE", "StrategyVersion": "V1"}
    assert ft.managing_version(position, active="ADAY-abc") == "V1"
    assert ft.managing_version({"Coin": "ETH", "Status": "ACTIVE"}, active="ADAY-abc") == "V1"


def test_ac77_new_buy_is_stamped_with_active_version(store):
    """AC77 — Yeni gerçek alış, giriş anındaki aktif strateji sürümüyle kaydedilir."""
    from position_journal import PositionJournal
    from trade_confirmation import confirm_buy

    portfolio = {"balance": 10000.0, "positions": []}
    executed = datetime(2026, 10, 3, 8, 0, tzinfo=timezone.utc)
    outcome = confirm_buy(portfolio, PositionJournal(), lambda: True, coin="Bitcoin (BTC)",
                          symbol="BTC-USD", quantity="0.1", price="100", stop="90",
                          executed_at=executed, now=executed + timedelta(minutes=5))
    assert outcome.ok, outcome
    assert portfolio["positions"][0]["StrategyVersion"] == "V1"
