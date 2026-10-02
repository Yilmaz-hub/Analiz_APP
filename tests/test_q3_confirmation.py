"""Q3 (spec 0003): gerçek işlem teyidi — zaman/miktar alanı, kimlik, tekrar, yazma sırası."""
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import storage
import trade_confirmation as tc
from app_helpers import break_writes_only, make_app, texts
from position_journal import PositionJournal

UTC = timezone.utc
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
TRADE_AT = datetime(2026, 9, 10, 9, 30, tzinfo=UTC)
D = Decimal


def _portfolio(balance=5000.0):
    return {"balance": balance, "positions": []}


def _buy(portfolio, journal, save=None, **over):
    args = dict(coin="Bitcoin (BTC)", symbol="BTC-USD", quantity=D("2"), price=D("100"), stop=D("90"),
                executed_at=TRADE_AT, now=NOW)
    args.update(over)
    return tc.confirm_buy(portfolio, journal, save or (lambda: True), **args)


def _sell(portfolio, journal, position, save=None, **over):
    args = dict(position=position, symbol="BTC-USD", quantity=D("2"), price=D("110"),
                executed_at=TRADE_AT + timedelta(hours=1), now=NOW)
    args.update(over)
    return tc.confirm_sell(portfolio, journal, save or (lambda: True), **args)


def test_q3_the_same_confirmation_twice_is_a_duplicate_and_changes_nothing(store):
    portfolio, journal = _portfolio(), PositionJournal()
    assert _buy(portfolio, journal).ok
    again = _buy(portfolio, journal)
    assert (again.ok, again.code) == (False, "TEKRAR_TEYIT")
    assert len(portfolio["positions"]) == 1 and len(journal.trades) == 1
    assert portfolio["balance"] == 4800.0                       # yalnız bir kez düşüldü


def test_q3_a_different_trade_time_quantity_or_price_is_a_separate_trade(store):
    portfolio, journal = _portfolio(), PositionJournal()
    assert _buy(portfolio, journal).ok
    assert _buy(portfolio, journal, executed_at=TRADE_AT + timedelta(minutes=1)).ok
    assert _buy(portfolio, journal, quantity=D("3")).ok
    assert _buy(portfolio, journal, price=D("101")).ok
    assert len(journal.trades) == 4


def test_q3_event_id_derives_from_the_trade_not_the_clock_zone():
    istanbul = TRADE_AT.astimezone(timezone(timedelta(hours=3)))
    assert tc.make_event_id("BTC-USD", "BUY", D("2.0"), D("100.00"), TRADE_AT) == \
        tc.make_event_id("BTC-USD", "BUY", D("2"), D("100"), istanbul)
    assert tc.make_event_id("BTC-USD", "BUY", D("2"), D("100"), TRADE_AT) != \
        tc.make_event_id("BTC-USD", "SELL", D("2"), D("100"), TRADE_AT)


@pytest.mark.parametrize("over,code", [
    (dict(executed_at=NOW + timedelta(minutes=1)), "GELECEK_ZAMAN"),
    (dict(quantity=D("0")), "GECERSIZ_MIKTAR"),
    (dict(price=D("-1")), "GECERSIZ_FIYAT"),
    (dict(stop=D("100")), "GECERSIZ_STOP"),
    (dict(quantity=D("1000")), "YETERSIZ_BAKIYE"),
])
def test_q3_invalid_buy_is_rejected_before_balance_or_position_change(store, over, code):
    portfolio, journal = _portfolio(), PositionJournal()
    result = _buy(portfolio, journal, **over)
    assert (result.ok, result.code) == (False, code)
    assert portfolio == {"balance": 5000.0, "positions": []} and journal.trades == []


def test_q3_portfolio_is_written_before_the_journal(store):
    order = []
    portfolio, journal = _portfolio(), PositionJournal()
    real = journal.confirm_trade
    journal.confirm_trade = lambda *a, **k: (order.append("journal"), real(*a, **k))[1]
    assert _buy(portfolio, journal, save=lambda: (order.append("portfolio"), True)[1]).ok
    assert order == ["portfolio", "journal"]


def test_q3_failed_portfolio_write_never_touches_the_journal(store):
    portfolio, journal = _portfolio(), PositionJournal()
    result = _buy(portfolio, journal, save=lambda: False)
    assert (result.ok, result.code) == (False, "KAYIT_YAZILAMADI") and journal.trades == []


def test_q3_interrupted_journal_write_is_completed_by_repeating_the_same_confirmation(store):
    portfolio, journal = _portfolio(), PositionJournal()
    geri_al = break_writes_only(store)
    try:
        first = _buy(portfolio, journal)
    finally:
        geri_al()
    assert (first.ok, first.code) == (False, "GUNLUK_YAZILAMADI")
    assert len(portfolio["positions"]) == 1 and journal.trades == []

    retry = _buy(portfolio, journal)
    assert retry.ok
    assert len(portfolio["positions"]) == 1 and len(journal.trades) == 1
    assert portfolio["balance"] == 4800.0
    assert _buy(portfolio, journal).code == "TEKRAR_TEYIT"      # artık gerçek tekrar


def test_q3_reconcile_completes_a_half_finished_trade_from_the_portfolio(store):
    portfolio, journal = _portfolio(), PositionJournal()
    geri_al = break_writes_only(store)
    try:
        _buy(portfolio, journal)
    finally:
        geri_al()
    assert tc.reconcile(portfolio, journal, {"Bitcoin (BTC)": "BTC-USD"}) == 1
    assert PositionJournal().positions["BTC-USD"].quantity == D("2")
    assert tc.reconcile(portfolio, journal, {"Bitcoin (BTC)": "BTC-USD"}) == 0


def test_q3_sell_closes_the_position_and_is_journaled(store):
    portfolio, journal = _portfolio(), PositionJournal()
    _buy(portfolio, journal)
    position = portfolio["positions"][0]
    assert _sell(portfolio, journal, position).ok
    assert position["Status"] == "CLOSED_CONFIRMED" and position["Realized"] == 20.0
    assert portfolio["balance"] == 4800.0 + 220.0
    assert "BTC-USD" not in PositionJournal().positions


@pytest.mark.parametrize("over,code", [
    (dict(quantity=D("1")), "MIKTAR_POZISYONLA_ESIT_DEGIL"),
    (dict(executed_at=NOW + timedelta(hours=1)), "GELECEK_ZAMAN"),
    (dict(executed_at=TRADE_AT - timedelta(hours=1)), "CIKIS_GIRISTEN_ONCE"),
    (dict(price=D("0")), "GECERSIZ_FIYAT"),
])
def test_q3_invalid_sell_changes_nothing(store, over, code):
    portfolio, journal = _portfolio(), PositionJournal()
    _buy(portfolio, journal)
    snapshot = repr(portfolio)
    result = _sell(portfolio, journal, portfolio["positions"][0], **over)
    assert (result.ok, result.code) == (False, code)
    assert repr(portfolio) == snapshot and len(journal.trades) == 1


# ---- Uygulama düzeyi -----------------------------------------------------------
def _hazirla(store):
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 50000.0, "positions": []})


def _tikla(at, etiket):
    for b in at.button:
        if b.label == etiket:
            return b.click().run()
    raise AssertionError(etiket)


def test_q3_app_buy_form_has_time_and_quantity_fields_and_rejects_future_time(store, monkeypatch, processed_df):
    _hazirla(store)
    at = make_app(monkeypatch, processed_df).run()
    assert not at.exception
    assert at.number_input(key="buy_qty:BTC-USD") and at.date_input(key="buy_date") and at.time_input(key="buy_time")
    at.date_input(key="buy_date").set_value(date.today() + timedelta(days=2)).run()
    at = _tikla(at, "➕ Emri Gir / Ekle")
    assert not at.exception
    assert "İşlem zamanı gelecekte olamaz" in texts(at)
    assert store.read_doc(store.PORTFOLIO_KEY) == {"balance": 50000.0, "positions": []}


def test_q3_app_buy_then_identical_buy_is_rejected_as_duplicate(store, monkeypatch, processed_df):
    _hazirla(store)
    at = make_app(monkeypatch, processed_df).run()
    at.number_input(key="buy_qty:BTC-USD").set_value(1.0)
    at.date_input(key="buy_date").set_value(date.today() - timedelta(days=1))
    at.time_input(key="buy_time").set_value(time(10, 0))
    at = at.run()
    stop = at.number_input[[n.label for n in at.number_input].index("Kuruma koyduğum stop")]
    stop.set_value(float(processed_df["Close"].iloc[-1]) * 0.5).run()
    at = _tikla(at, "➕ Emri Gir / Ekle")
    assert not at.exception
    assert len(store.read_doc(store.PORTFOLIO_KEY)["positions"]) == 1
    first_balance = store.read_doc(store.PORTFOLIO_KEY)["balance"]

    at = _tikla(at, "➕ Emri Gir / Ekle")
    assert "daha önce teyit edilmiş" in texts(at)
    after = store.read_doc(store.PORTFOLIO_KEY)
    assert len(after["positions"]) == 1 and after["balance"] == first_balance
    assert len(PositionJournal().trades) == 1
