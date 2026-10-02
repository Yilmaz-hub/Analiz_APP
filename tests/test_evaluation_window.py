"""Spec 0005 Adım 4 / S2 — dönem bölme, al-tut ve eşit karşılaştırma (AC11–14, 47, 53, 54, 56, 57, 62)."""
from datetime import date, timedelta
from decimal import Decimal

from evaluation_window import (
    EvaluationInput, buy_and_hold, comparable, select_period, split_dates,
)
from trade_execution import CostAssumptions

D0 = date(2026, 1, 1)
ZERO_COSTS = CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0"))


def _dates(n, start=D0):
    return [start + timedelta(days=i) for i in range(n)]


def _input(capital="10000", dates=None, external_cash=False):
    days = dates if dates is not None else _dates(100)
    return EvaluationInput(Decimal(capital), days[0], days[-1], days, external_cash)


def test_ac11_unequal_start_capital_is_not_comparable():
    """AC11 — 10.000 ile 10.000,01 başlangıç sermayesi eşit koşul sayılmaz; üstünlük sonucu yayımlanmaz."""
    result = comparable(_input("10000"), _input("10000.01"))
    assert not result.ok
    assert "eşit" in result.reason


def test_ac11_unequal_period_is_not_comparable():
    """AC11 — Dönemi tam eşit olmayan iki değerlendirme karşılaştırılamaz."""
    assert not comparable(_input(dates=_dates(100)), _input(dates=_dates(99))).ok


def test_ac11_equal_conditions_are_comparable():
    """AC11 — Sermaye ve dönem tam eşitse karşılaştırma uygundur."""
    assert comparable(_input("10000.00"), _input("10000")).ok


def test_ac12_future_prices_do_not_change_past_decisions():
    """AC12 — Karar tarihinden sonraki fiyatlar değiştirilince o tarihe kadarki kararlar değişmez."""
    from conftest import make_ohlcv
    from data_fetchers import process_data
    from technical_analysis import build_v1_decisions

    raw = make_ohlcv()
    cut = 350
    changed = raw.copy()
    changed.iloc[cut + 1:, :4] = changed.iloc[cut + 1:, :4].values * 1.7
    base, _ = process_data(raw, "test")
    other, _ = process_data(changed, "test")
    before = build_v1_decisions(base, include_ml=False)
    after = build_v1_decisions(other, include_ml=False)
    cutoff = raw.index[cut]
    checked = [day for day in before if day <= cutoff]
    assert len(checked) > 100
    assert all(before[day] == after[day] for day in checked)


def test_ac13_hundred_dates_split_sixty_forty():
    """AC13 — 100 uygun tarihte ilk 60 ayar seçimine, kalan 40 değerlendirmeye gider."""
    days = _dates(100)
    split = split_dates(days)
    assert split.tuning == days[:60]
    assert split.evaluation == days[60:]


def test_ac14_external_cash_flow_makes_comparison_unsuitable():
    """AC14 — Dönemde dışarıdan para yatırılmışsa strateji getiri karşılaştırması uygun sayılmaz."""
    result = comparable(_input(), _input(external_cash=True))
    assert not result.ok
    assert "nakit" in result.reason


def test_ac47_buy_and_hold_final_equity():
    """AC47 — Sıfır maliyet, 1.000 sermaye, açılış 100, adım 1, son fiyat 110 → son sermaye 1.100."""
    result = buy_and_hold(Decimal("1000"), Decimal("100"), Decimal("110"), Decimal("1"), ZERO_COSTS)
    assert result.executed
    assert result.final_equity == Decimal("1100")


def test_ac53_seven_dates_split_four_three():
    """AC53 — 7 uygun tarihte ayar seçimine 4, ayrı değerlendirmeye 3 tarih atanır (aşağı yuvarla)."""
    days = _dates(7)
    split = split_dates(days)
    assert (len(split.tuning), len(split.evaluation)) == (4, 3)
    assert split.tuning + split.evaluation == days


def test_ac54_249_dates_is_insufficient_history_250_is_enough():
    """AC54 — 249 uygun tarih "yetersiz geçmiş" sayılır; 250 yeterli."""
    assert split_dates(_dates(249)).sufficient_history is False
    assert split_dates(_dates(250)).sufficient_history is True


def test_ac56_no_common_dates_blocks_comparison_with_reason():
    """AC56 — Ortak işlem tarihi olmayan iki varlık karşılaştırılmaz, neden gösterilir."""
    a = _input(dates=_dates(10))
    b = _input(dates=_dates(10, start=D0 + timedelta(days=100)))
    # dönem eşitliği ayrıca sınanmasın: aynı başlangıç/bitiş etiketi ver
    b = EvaluationInput(b.initial_capital, a.start, a.end, b.dates, False)
    result = comparable(a, b)
    assert not result.ok
    assert "ortak" in result.reason.lower()


def test_ac57_period_before_first_data_gives_insufficient_history():
    """AC57 — Dönemin tümü varlığın ilk verisinden önceyse metrik üretilmez, "yetersiz geçmiş" görünür."""
    asset_days = _dates(30, start=D0 + timedelta(days=100))
    selected = select_period(asset_days, D0, D0 + timedelta(days=20))
    assert selected.dates == []
    assert selected.insufficient_history
    assert "yetersiz geçmiş" in selected.reason.lower()


def test_ac62_buy_and_hold_leftover_cash_counts_in_final_equity():
    """AC62 — 1.000 sermaye, açılış 300, adım 1: 3 birim alınır, kalan 100 nakit son sermayeye girer."""
    result = buy_and_hold(Decimal("1000"), Decimal("300"), Decimal("300"), Decimal("1"), ZERO_COSTS)
    assert result.quantity == Decimal("3")
    assert result.leftover_cash == Decimal("100")
    assert result.final_equity == Decimal("1000")


def test_buy_and_hold_unknown_commission_is_marked_upper_bound():
    """AC84 — Komisyonu bilinmeyen al-tut sonucu üst sınır olarak etiketlenir, sıfır maliyet sayılmaz."""
    unknown = CostAssumptions(Decimal("0"), Decimal("0"), None)
    assert buy_and_hold(Decimal("1000"), Decimal("100"), Decimal("110"), Decimal("1"),
                        unknown).is_upper_bound
    assert not buy_and_hold(Decimal("1000"), Decimal("100"), Decimal("110"), Decimal("1"),
                            ZERO_COSTS).is_upper_bound


def test_buy_and_hold_commission_reduces_affordable_quantity():
    """AC47 — Bilinen komisyon alınabilen miktarı düşürür (maliyet sonrası alınabilen miktar)."""
    costs = CostAssumptions(Decimal("0"), Decimal("0"), Decimal("1"))  # %1
    result = buy_and_hold(Decimal("1000"), Decimal("100"), Decimal("100"), Decimal("1"), costs)
    assert result.quantity == Decimal("9")            # 10 adet 1.010 ister, nakit 1.000
    assert result.leftover_cash == Decimal("1000") - Decimal("900") - Decimal("9")


def test_buy_and_hold_unaffordable_or_unknown_step_is_not_executed():
    """AC47 — Miktar adımı bilinmiyorsa ya da adım alınamıyorsa alım yapılmaz ve neden görünür."""
    unknown_step = buy_and_hold(Decimal("1000"), Decimal("100"), Decimal("110"), None, ZERO_COSTS)
    assert not unknown_step.executed and unknown_step.reason
    too_pricey = buy_and_hold(Decimal("50"), Decimal("100"), Decimal("110"), Decimal("1"), ZERO_COSTS)
    assert not too_pricey.executed and too_pricey.reason
