"""Spec 0005 Adım 6 / S5 — sanal kâr koruma adayları (R13, R14, Q05)."""
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import profit_protection as pp
from trade_execution import Bar, evaluate_stop

D = Decimal
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _bar(day, o, h, l, c):
    return Bar(T0 + timedelta(days=day), D(str(o)), D(str(h)), D(str(l)), D(str(c)))


def _position(qty="10", entry="100", stop="90"):
    return pp.VirtualPosition(entry=D(entry), quantity=D(qty), stop=D(stop), entry_at=T0)


def test_ac32_virtual_trailing_stop_never_changes_real_stop():
    """AC32 — Sanal iz süren stop ilerletilince gerçek pozisyondaki kayıtlı stop değişmez."""
    portfolio = {"positions": [{"Coin": "ETH", "Giriş": 100.0, "Adet": 10.0, "Stop": 90.0,
                                "Gerçekleşme Zamanı": T0.isoformat()}]}
    before = copy.deepcopy(portfolio)
    position = pp.from_portfolio(portfolio["positions"][0])
    bars = [(_bar(i, 100 + 5 * i, 102 + 5 * i, 99 + 5 * i, 101 + 5 * i), D("2")) for i in range(1, 6)]
    result = pp.simulate(position, bars, pp.TRAILING, quantity_step=D("1"))
    assert result.stop_path[-1] > D("90")  # sanal stop ilerledi
    assert portfolio == before            # gerçek kayıt aynen kaldı


def test_ac33_touch_strictly_before_raise_effective_time_does_not_trigger_new_stop():
    """AC33 — Stop yükseltmesinin geçerlilik anından kesin önceki temas yeni stopla satış oluşturmaz."""
    bar = _bar(1, 100, 101, 94, 99)
    assert evaluate_stop(bar, D("95"), active_at=bar.at + timedelta(seconds=1)) is None


def test_ac59_touch_at_exact_effective_time_counts_for_new_stop():
    """AC59 — Temas geçerlilik anıyla tam aynı zamandaysa yeni stopa dahil sayılır (≥)."""
    bar = _bar(1, 100, 101, 94, 99)
    fill = evaluate_stop(bar, D("95"), active_at=bar.at)
    assert fill is not None and fill.base_price == D("95")


def test_ac34_partial_exit_keeps_rest_open_without_closing_position():
    """AC34 — 10 birimin 4'ü satılınca 6 birim açık kalır, kapanmış pozisyon sayısı artmaz."""
    book = pp.VirtualBook(_position(), quantity_step=D("1"))
    book.sell(D("4"), D("110"), T0 + timedelta(days=2), "KADEME")
    assert book.open_quantity == D("6")
    assert book.closed_positions == 0


def test_ac61_last_part_closes_position_exactly_once():
    """AC61 — Kademeli çıkışta son parça da kapanınca kapanmış işlem sayısı tam 1 artar."""
    book = pp.VirtualBook(_position(), quantity_step=D("1"))
    book.sell(D("4"), D("110"), T0 + timedelta(days=2), "KADEME")
    book.sell(D("6"), D("105"), T0 + timedelta(days=3), "STOP")
    assert book.open_quantity == D("0")
    assert book.closed_positions == 1


def test_ac60_fractional_partial_leaves_single_step_aligned_remaining():
    """AC60 — Adım 0,0001 olan varlıkta kısmi satış sonrası kalan miktar adıma uygun tek değerdir."""
    book = pp.VirtualBook(_position(qty="0.3333"), quantity_step=D("0.0001"))
    sold = book.sell_fraction(D("0.5"), D("110"), T0 + timedelta(days=2), "KADEME")
    assert sold == D("0.1666")
    assert book.open_quantity == D("0.1667")
    assert pp.format_quantity(book.open_quantity, D("0.0001")) == "0.1667"


def test_target_candidate_sells_all_at_2r_from_next_day():
    """Q05 — Sabit hedef: giriş + 2R'de tamamı satılır; seviye ertesi gün açılışından geçerlidir."""
    bars = [(_bar(1, 101, 115, 100, 112), D("2")), (_bar(2, 113, 125, 112, 121), D("2"))]
    result = pp.simulate(_position(), bars, pp.TARGET, quantity_step=D("1"))
    assert [(f.quantity, f.price, f.reason) for f in result.fills] == [(D("10"), D("120"), "HEDEF")]
    assert result.closed_positions == 1


def test_trailing_candidate_only_moves_up_and_exits_on_touch():
    """Q05 — İz süren stop: en yüksek kapanış − 2×ATR, yalnız yukarı; temas olunca çıkış."""
    bars = [
        (_bar(1, 101, 111, 100, 110), D("2")),  # stop → 106 (ertesi gün geçerli)
        (_bar(2, 109, 109, 107, 108), D("2")),  # 104 < 106 → stop 106'da kalır
        (_bar(3, 107, 108, 105, 106), D("2")),  # düşük 105 ≤ 106 → çıkış 106
    ]
    result = pp.simulate(_position(), bars, pp.TRAILING, quantity_step=D("1"))
    assert result.stop_path == [D("90"), D("106"), D("106")]
    assert [(f.price, f.reason) for f in result.fills] == [(D("106"), "STOP")]


def test_scale_out_candidate_sells_half_at_1r_rest_by_stop():
    """Q05 — Kademeli çıkış: giriş + 1R'de yarısı satılır, kalan V1 stopu ile çıkar."""
    bars = [(_bar(1, 101, 111, 100, 108), D("2")), (_bar(2, 100, 101, 88, 89), D("2"))]
    result = pp.simulate(_position(), bars, pp.SCALE_OUT, quantity_step=D("1"))
    assert [(f.quantity, f.price, f.reason) for f in result.fills] == [
        (D("5"), D("110"), "KADEME"), (D("5"), D("90"), "STOP")]
    assert result.closed_positions == 1


def test_v1_sell_signal_closes_remaining_at_next_open():
    """R13 — Referans çıkış korunur: SAT uyarısı ertesi açılışta kalan miktarı kapatır."""
    bars = [(_bar(1, 101, 105, 100, 104), D("2")), (_bar(2, 103, 104, 101, 102), D("2"))]
    result = pp.simulate(_position(), bars, pp.TARGET, quantity_step=D("1"),
                         sell_signals={bars[0][0].at})
    assert [(f.price, f.reason) for f in result.fills] == [(D("103"), "SAT")]
