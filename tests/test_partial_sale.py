"""Spec 0006 A — kısmi satış teyidi (R01–R08, Q01–Q06)."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import trade_confirmation as tc
from position_journal import PositionJournal

UTC = timezone.utc
NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
TRADE_AT = datetime(2026, 9, 10, 9, 30, tzinfo=UTC)
D = Decimal


def _portfolio(balance=5000.0):
    return {"balance": balance, "positions": []}


def _buy(portfolio, journal, **over):
    args = dict(coin="Bitcoin (BTC)", symbol="BTC-USD", quantity=D("10"), price=D("100"), stop=D("90"),
                executed_at=TRADE_AT, now=NOW)
    args.update(over)
    return tc.confirm_buy(portfolio, journal, lambda: True, **args)


def _sell(portfolio, journal, quantity, **over):
    args = dict(position=portfolio["positions"][0], symbol="BTC-USD", quantity=D(str(quantity)),
                price=D("110"), executed_at=TRADE_AT + timedelta(hours=1), now=NOW)
    args.update(over)
    return tc.confirm_sell(portfolio, journal, lambda: True, **args)


@pytest.fixture
def held(store):
    portfolio, journal = _portfolio(), PositionJournal()
    assert _buy(portfolio, journal).ok
    return portfolio, journal


def _closed(portfolio):
    return sum(1 for p in portfolio["positions"] if p["Status"] == "CLOSED_CONFIRMED")


def test_ac01_partial_quantity_is_accepted(held):
    """AC01 — 10 adetlik pozisyonun 4 adedi satış olarak teyit edildiğinde satış kabul edilir."""
    portfolio, journal = held
    assert _sell(portfolio, journal, 4).ok


def test_ac02_position_stays_active_with_remaining_quantity(held):
    """AC02 — Kısmi satıştan sonra pozisyon aktif kalır ve kalan miktar tam 6'dır."""
    portfolio, journal = held
    _sell(portfolio, journal, 4)
    position = portfolio["positions"][0]
    assert position["Status"] == "ACTIVE" and position["Adet"] == 6.0


def test_ac03_fractional_remaining_has_no_float_residue(store):
    """AC03 — 0,3333 adetlik pozisyonda 0,1666 satıldığında kalan tam 0,1667'dir (kayan nokta artığı yok)."""
    portfolio, journal = _portfolio(), PositionJournal()
    assert _buy(portfolio, journal, quantity=D("0.3333")).ok
    assert _sell(portfolio, journal, "0.1666").ok
    remaining = portfolio["positions"][0]["Adet"]
    assert Decimal(str(remaining)) == D("0.1667") and repr(remaining) == "0.1667"


def test_ac04_full_quantity_still_closes_the_position(held):
    """AC04 — Eldeki miktarın tamamı satıldığında pozisyon bugünkü gibi kapanır ve kalan miktar 0'dır."""
    portfolio, journal = held
    assert _sell(portfolio, journal, 10).ok
    position = portfolio["positions"][0]
    assert position["Status"] == "CLOSED_CONFIRMED" and position["Adet"] == 0.0
    assert "BTC-USD" not in PositionJournal().positions


def test_ac05_more_than_held_is_rejected_and_changes_nothing(held):
    """AC05 — Eldeki miktardan büyük satış reddedilir, nedeni gösterilir ve kayıt değişmez."""
    portfolio, journal = held
    snapshot, trades = repr(portfolio), len(journal.trades)
    result = _sell(portfolio, journal, 11)
    assert (result.ok, result.code) == (False, "MIKTAR_FAZLA")
    assert repr(portfolio) == snapshot and len(journal.trades) == trades
    from trading_ui import describe_code
    assert "eldeki miktardan fazla" in describe_code("MIKTAR_FAZLA")


@pytest.mark.parametrize("quantity", ["0", "-1"])
def test_ac06_zero_or_negative_quantity_is_rejected(held, quantity):
    """AC06 — 0 veya negatif miktarlı satış reddedilir ve kayıt değişmez."""
    portfolio, journal = held
    snapshot = repr(portfolio)
    result = _sell(portfolio, journal, quantity)
    assert (result.ok, result.code) == (False, "GECERSIZ_MIKTAR")
    assert repr(portfolio) == snapshot


def test_ac07_stop_is_unchanged_after_partial_sale(held):
    """AC07 — Kısmi satıştan sonra kayıtlı stop aynen kalır."""
    portfolio, journal = held
    _sell(portfolio, journal, 4)
    assert portfolio["positions"][0]["Stop"] == 90.0


def test_ac08_realized_cash_and_invested_follow_the_sold_part(held):
    """AC08 — Giriş 100, satış 110, 4 adet: kâr 40, nakit 440 artar, yatırım kalan 6 adedin maliyeti 600'e iner."""
    portfolio, journal = held
    cash_before = portfolio["balance"]
    _sell(portfolio, journal, 4)
    position = portfolio["positions"][0]
    assert position["Realized"] == 40.0
    assert portfolio["balance"] == cash_before + 440.0
    assert position["Yatırım"] == 600.0


def test_ac09_closed_trade_is_counted_only_when_last_part_closes(held):
    """AC09 — İki parça halinde satılan pozisyon, ilk parçadan sonra kapanmış sayıyı artırmaz; son parçadan sonra tam 1 artırır."""
    portfolio, journal = held
    _sell(portfolio, journal, 4)
    assert _closed(portfolio) == 0
    assert _sell(portfolio, journal, 6, executed_at=TRADE_AT + timedelta(hours=2)).ok
    assert _closed(portfolio) == 1
    position = portfolio["positions"][0]
    assert position["Realized"] == 100.0 and position["Çıkış Adedi"] == 10.0
    assert [e["quantity"] for e in position["Çıkışlar"]] == ["4", "6"]


def test_ac10_percent_converts_to_quantity():
    """AC10 — %40 girildiğinde 10 adetlik pozisyon için satılacak miktar 4 olur."""
    assert tc.quantity_from_percent(D("10"), D("40"), None) == (D("4"), "")


def test_ac11_percent_rounds_down_never_up():
    """AC11 — Miktar adımı 1 olan 10 adetlik pozisyonda %45 girildiğinde satılacak miktar 4'tür."""
    assert tc.quantity_from_percent(D("10"), D("45"), D("1")) == (D("4"), "")
    assert tc.quantity_from_percent(D("10"), D("100"), D("3")) == (D("10"), "")  # tam miktar bölünmez


def test_ac12_percent_rounding_to_zero_is_rejected():
    """AC12 — Miktar adımı 1 olan 10 adetlik pozisyonda %5 girildiğinde (0,5 adet → 0) satış reddedilir ve nedeni gösterilir."""
    quantity, code = tc.quantity_from_percent(D("10"), D("5"), D("1"))
    assert quantity is None and code == "YUZDE_SIFIRA_DUSTU"
    from trading_ui import describe_code
    assert "miktar adımının altında" in describe_code(code)
    for bad in (D("0"), D("-5"), D("101")):
        assert tc.quantity_from_percent(D("10"), bad, None)[1] == "GECERSIZ_YUZDE"


def test_ac13_same_partial_sale_twice_creates_no_second_record(held):
    """AC13 — Aynı kısmi satış ikinci kez teyit edildiğinde ikinci kayıt oluşmaz, kalan miktar değişmez."""
    portfolio, journal = held
    assert _sell(portfolio, journal, 4).ok
    again = _sell(portfolio, journal, 4)
    assert (again.ok, again.code) == (False, "TEKRAR_TEYIT")
    assert portfolio["positions"][0]["Adet"] == 6.0
    assert len(journal.trades) == 2 and len(portfolio["positions"][0]["Çıkışlar"]) == 1


def test_ac14_journal_rebuilds_the_remaining_position(held):
    """AC14 — Kısmi satıştan sonra günlük, pozisyonu kalan miktarla yeniden kurar (pozisyon silinmez)."""
    portfolio, journal = held
    _sell(portfolio, journal, 4)
    reloaded = PositionJournal()
    assert reloaded.positions["BTC-USD"].quantity == D("6")
    assert reloaded.positions["BTC-USD"].entry_price == D("100")
    _sell(portfolio, journal, 6, executed_at=TRADE_AT + timedelta(hours=2))
    assert "BTC-USD" not in PositionJournal().positions


def test_ac15_execution_and_record_times_stay_separate(held):
    """AC15 — Dünkü kısmi satış bugün kaydedildiğinde işlem zamanı dün, kayıt zamanı bugün olarak korunur."""
    portfolio, journal = held
    executed = TRADE_AT + timedelta(hours=1)
    _sell(portfolio, journal, 4, executed_at=executed)
    trade = PositionJournal().trades[-1]
    assert trade.side == "SELL" and trade.executed_at == executed
    assert trade.recorded_at.date() > executed.date()


def test_ac16_legacy_single_exit_record_is_read_unchanged(store):
    """AC16 — Tek çıkış alanlı eski bir kapanmış kayıt hatasız okunur ve aynı sonucu gösterir."""
    legacy = {"Coin": "Bitcoin (BTC)", "Giriş": 100.0, "Adet": 0.0, "Yatırım": 0.0, "Realized": 20.0,
              "Status": "CLOSED_CONFIRMED", "V1Verified": True, "JournalEventId": "b1",
              "Gerçekleşme Zamanı": TRADE_AT.isoformat(), "Gerçekleşen Çıkış": 110.0,
              "Çıkış Adedi": 2.0, "Çıkış Zamanı": (TRADE_AT + timedelta(hours=1)).isoformat(),
              "JournalExitEventId": "s1"}
    journal = PositionJournal()
    assert tc.reconcile({"balance": 0.0, "positions": [legacy]}, journal, {"Bitcoin (BTC)": "BTC-USD"}) == 2
    assert [(t.side, t.quantity) for t in journal.trades] == [("BUY", D("2")), ("SELL", D("2"))]


def test_ac16_reconcile_rebuilds_a_partially_sold_position_from_exit_list(store):
    """AC16 — Çıkış listeli kısmi kayıt, günlükte eksik olan alış ve satışları doğru miktarlarla tamamlar."""
    portfolio, journal = _portfolio(), PositionJournal()
    _buy(portfolio, journal)
    _sell(portfolio, journal, 4)
    import storage
    storage.write_doc(storage.JOURNAL_KEY, {"trades": [], "cooldown_bars": 0})
    rebuilt = PositionJournal()
    assert tc.reconcile(portfolio, rebuilt, {"Bitcoin (BTC)": "BTC-USD"}) == 2
    assert [(t.side, t.quantity) for t in rebuilt.trades] == [("BUY", D("10")), ("SELL", D("4"))]
    assert rebuilt.positions["BTC-USD"].quantity == D("6")


def test_ac18_sell_signal_still_means_full_exit_and_nothing_is_automatic(held):
    """AC18 — SAT uyarısı "tam çıkış" yönlendirmesi olarak kalır; kısmi satış kendiliğinden yapılmaz."""
    from trade_decisions import decide_action
    from trading_contracts import Action, Position, PositionState, Signal

    portfolio, journal = held
    snapshot = repr(portfolio)
    open_position = Position(PositionState.OPEN, D("10"), D("100"), TRADE_AT, D("90"))
    assert decide_action(Signal.SELL, open_position).action is Action.SELL
    assert repr(portfolio) == snapshot  # karar üretmek kayda dokunmaz
