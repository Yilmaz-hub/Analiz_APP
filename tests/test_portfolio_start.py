"""Spec 0025 — portföy başlangıcı (BTC/ETH), ısınmada işlem yok, geç gelen varlığın payı nakit; nakit yetmezse
eldeki nakitle alım."""
from decimal import Decimal

import pandas as pd

import portfolio_backtest as pb
from trade_execution import CostAssumptions

COSTS = CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0.1"))
T = pd.Timestamp


def _flat(n=10, price=100.0, start="2026-01-01"):
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1, "Close": price, "ATR": 1.0}, index=idx)


def test_ac01_start_follows_btc_eth_and_later_assets_are_listed():
    """AC01 — Başlangıç BTC/ETH'nin en erken işlem günüdür; daha erken hazır olan (altın) beklenmez, daha geç hazır olanlar listelenir; referans yoksa en erken gün."""
    ready = {"BTC": T("2024-08-05"), "ETH": T("2024-08-06"), "Altın": T("2023-08-02"), "RENDER": T("2025-02-11")}
    start, late = pb.portfolio_start(ready, ["BTC", "ETH"])
    assert start == T("2024-08-05") and late == {"ETH": T("2024-08-06"), "RENDER": T("2025-02-11")}
    start, late = pb.portfolio_start({"Altın": T("2023-08-02"), "SOL": T("2024-08-05")}, [])
    assert start == T("2023-08-02") and late == {"SOL": T("2024-08-05")}


def test_ac02_late_asset_share_stays_cash_and_hold_buys_on_its_first_day():
    """AC02 — Geç hazır olan varlığın payı o güne kadar nakittir: al-tut ve tut alımı o gün yapılır, öncesinde nakit değişmez."""
    a, b = _flat(), _flat()
    ready = {"A": a.index[0], "B": b.index[5]}
    result = pb.run_portfolio({"A": a, "B": b}, {"A": {}, "B": {}}, {"A": Decimal("50"), "B": Decimal("50")},
                              capital=Decimal("10000"), costs=COSTS,
                              quantity_steps={"A": Decimal("0.0001"), "B": Decimal("0.0001")},
                              hold={"A", "B"}, available_from=ready)
    assert abs(result.holdings["B"] - Decimal("4995")) < 10 and abs(result.holdings["A"] - Decimal("4995")) < 10
    hold = dict(result.hold_equity)
    before, after = hold[a.index[4].date()], hold[a.index[5].date()]
    assert abs(before - Decimal("9995")) < 1                     # yalnız A alınmış (komisyon kadar eksik)
    assert abs(after - Decimal("9990")) < 1                      # B o gün alındı


def _series(n, start, seed):
    from conftest import make_ohlcv
    from data_fetchers import process_data

    raw = make_ohlcv(seed=seed, segments=[(n, 0.002)])
    raw.index = pd.date_range(start, periods=n, freq="D", tz="UTC")
    frame, _ = process_data(raw, "fixture")
    return frame


class _St:
    def __init__(self):
        self.warnings = []

    def progress(self, *a):
        return self

    def empty(self):
        pass

    def warning(self, text):
        self.warnings.append(text)


def test_ac03_panel_starts_with_btc_warns_for_late_assets_and_never_trades_in_warm_up(monkeypatch):
    """AC03 — Portföy testi BTC'nin ilk işlem gününden başlar; daha uzun geçmişli altın beklenmez; geç başlayan
    altcoin için "o tarihe kadar payı nakit sayıldı" uyarısı çıkar; hiçbir varlık ısınma döneminde işlem yapmaz."""
    import data_fetchers
    import portfolio_ui
    from technical_analysis import build_v1_decisions

    frames = {"BTC-USD": _series(420, "2024-01-01", 1), "XAU_GOLD": _series(520, "2023-09-23", 2),
              "SOL-USD": _series(320, "2024-04-10", 3)}
    monkeypatch.setattr(data_fetchers, "get_market_data", lambda src, symbol, tf: (frames[symbol], "fixture"))
    names = {"Bitcoin": "BTC-USD", "Altın": "XAU_GOLD", "Solana": "SOL-USD"}
    weights = {"Bitcoin": Decimal("40"), "Altın": Decimal("30"), "Solana": Decimal("30")}
    st = _St()
    result, _, methods = portfolio_ui._run(list(names), weights, names, "Binance", Decimal("10000"), Decimal("0.1"), st)
    ready = {name: pb.first_tradable_day(frames[s], list(build_v1_decisions(frames[s]))) for name, s in names.items()}
    assert methods["Solana"] == "KANAL"
    assert result.start == ready["Bitcoin"].date() and result.hold_equity[0][0] == ready["Bitcoin"].date()
    assert len(st.warnings) == 1 and "Solana" in st.warnings[0] and "nakit sayıldı" in st.warnings[0]
    assert str(ready["Solana"].date()) in st.warnings[0] and "%30" in st.warnings[0]
    for trade in result.trades:
        assert trade.entry_at >= ready[trade.symbol], trade


def test_ac04_single_backtest_buys_with_remaining_cash_after_a_loss():
    """AC04 — Tek varlık backtest'inde işlem tutarı ≈ sermaye iken zararlı işlemden sonra kalan nakitle alım sürer;
    nakit tutarın yarısından azsa alım "nakit yetersiz" olarak engellenir."""
    from technical_analysis import run_v1_strategy_backtest

    frame = _flat(n=20)
    frame.iloc[4, frame.columns.get_loc("Low")] = 90.0          # 2,5 ATR stop: zararlı çıkış
    dec = {ts: "AL" for ts in frame.index}
    run = dict(initial_cash=Decimal("10000"), quantity_step=Decimal("0.0001"), costs=COSTS)
    full = run_v1_strategy_backtest(frame, dec, trade_notional=Decimal("9900"), **run)
    assert len(full["trades"]) >= 1 and full["position"] is not None   # stop sonrası kalan nakitle yeniden girdi
    assert "YETERSIZ_NAKIT" not in full["blocked"]
    tiny = run_v1_strategy_backtest(frame, dec, trade_notional=Decimal("25000"), **run)
    assert tiny["blocked"].get("YETERSIZ_NAKIT", 0) >= 1 and not tiny["trades"]
