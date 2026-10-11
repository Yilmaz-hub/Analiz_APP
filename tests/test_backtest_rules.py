"""Spec 0023 — tek varlık backtest'i varlığın kuralıyla (BTC/ETH/altın C, altcoin kanal, hisse al-tut)."""
from decimal import Decimal

import portfolio_backtest as pb
import signal_rules as sr
import strategy_candidates as sc
from trade_execution import CostAssumptions

COSTS = CostAssumptions(Decimal("0"), Decimal("0"), Decimal("0.1"))
WARM_UP = 199


def _v1(frame):
    """V1'e benzer karar sözlüğü: ilk karar 199. mumda (ısınma), arada AL ve SAT günleri."""
    idx = frame.index
    decisions = {ts: "BEKLE" for ts in idx[WARM_UP:]}
    for pos in range(WARM_UP, len(idx), 15):
        decisions[idx[pos]] = "AL"
    for pos in range(WARM_UP + 7, len(idx), 30):
        decisions[idx[pos]] = "SAT"
    return decisions


# ---- AC01: kural kararları ----------------------------------------------------------------------
def test_ac01_rule_decisions_follow_the_asset_rule_from_the_first_v1_day(processed_df):
    """AC01 — BTC C kararlarını (V1 AL + ortalama üstü), SOL kanal kararlarını yalnız ilk V1 karar gününden itibaren; döviz ve "V1" modunda V1 kararlarını aynen döndürür."""
    frame, v1 = processed_df, _v1(processed_df)
    first = frame.index[WARM_UP]

    trend, method = sr.rule_decisions("BTC-USD", frame, v1)
    assert method == "TREND_SMA" and min(trend) == first and set(trend) == set(v1)
    sma = frame["Close"].rolling(200).mean()
    for day, verdict in trend.items():
        above = Decimal(str(frame["Close"][day])) > Decimal(str(sma[day]))
        expected = "SAT" if v1[day] == "SAT" else "AL" if v1[day] == "AL" and above else "BEKLE"
        assert verdict == expected
    assert "AL" in trend.values() and "AL" in [v for d, v in v1.items() if trend[d] == "BEKLE"]

    channel, method = sr.rule_decisions("SOL-USD", frame, v1)
    full = sc.candidate_decisions(frame, {}, sc.channel_candidate("KRIPTO"), None)
    assert method == "KANAL" and min(channel) == first
    assert channel == {day: verdict for day, verdict in full.items() if day >= first}
    assert sr.rule_decisions("SOL-USD", frame, {**v1, first: "SAT"})[0] == channel     # V1 kanalı etkilemez

    assert sr.rule_decisions("EURUSD=X", frame, v1) == (v1, None)
    sr.choose_signal_mode(sr.V1)
    assert sr.rule_decisions("BTC-USD", frame, v1) == (v1, None)
    assert sr.rule_decisions("SOL-USD", frame, v1) == (v1, None)


# ---- AC05: motor eşliği -------------------------------------------------------------------------
def test_ac05_channel_decisions_give_the_same_trades_in_both_engines(processed_df):
    """AC05 — Kanal kararlarıyla tek varlık motoru ve %100 ağırlıklı portföy motoru aynı giriş/çıkış günlerini üretir."""
    from technical_analysis import run_v1_strategy_backtest

    frame = processed_df
    decisions, method = sr.rule_decisions("SOL-USD", frame, _v1(frame))
    assert method == "KANAL"
    single = run_v1_strategy_backtest(frame, decisions, initial_cash=Decimal("10000"),
                                      trade_notional=Decimal("9000"), quantity_step=Decimal("0.0001"), costs=COSTS)
    multi = pb.run_portfolio({"X": frame}, {"X": decisions}, {"X": Decimal("100")}, capital=Decimal("10000"),
                             costs=COSTS, quantity_steps={"X": Decimal("0.0001")})
    assert single["trades"]
    assert [(t["entry_at"], t["exit_at"], t["reason"]) for t in single["trades"]] == \
           [(t.entry_at, t.exit_at, t.reason) for t in multi.trades]


# ---- Ekran (AC02–AC04) --------------------------------------------------------------------------
def _screen(store, monkeypatch, frame, name, symbol, v1=None):
    import technical_analysis
    from app_helpers import make_app

    monkeypatch.setattr(technical_analysis, "build_v1_decisions", lambda df, **k: v1 or _v1(df))
    store.write_doc(store.ASSETS_KEY, {name: symbol})
    at = make_app(monkeypatch, frame).run()
    at = next(sb for sb in at.selectbox if sb.label == "Periyot:").set_value("1d").run()
    at.text_input(key=f"ts:{symbol}:quantity_step").set_value("0.001")
    return at.run()


def _infos(at):
    return [str(i.value) for i in at.info]


def test_ac02_stocks_show_only_the_hold_result_without_a_timing_backtest(store, monkeypatch, processed_df):
    """AC02 — AAPL / THYAO.IS'de zamanlama backtest'i çalışmaz; ekranda "al-tut" bilgisi ve al-tut sonucu görünür."""
    import technical_analysis
    from app_helpers import click

    def no_timing_test(*a, **k):
        raise AssertionError("hissede zamanlama backtest'i çalıştı")

    monkeypatch.setattr(technical_analysis, "run_v1_strategy_backtest", no_timing_test)
    for name, symbol in (("Apple (AAPL)", "AAPL"), ("Türk Hava Yolları (THYAO)", "THYAO.IS")):
        at = click(_screen(store, monkeypatch, processed_df, name, symbol), "🚀 Backtest Başlat")
        assert not at.exception
        assert "Bu varlıkta yöntem al-tut; zamanlama testi yok." in _infos(at)
        labels = [m.label for m in at.metric]
        assert "Varlığın fiyat değişimi" in labels and "Toplam Getiri" not in labels
        assert "perf_report_source" not in at.session_state
        assert not any(s.value == "📑 Güvenilir Performans Raporu" for s in at.subheader)


def test_ac03_backtest_label_names_the_rule_and_v1_mode_keeps_the_old_text(store, monkeypatch, processed_df):
    """AC03 — BTC günlük backtest bilgi kutusu kural adını ("V1 + 200 günlük ortalama") yazar; "V1 (eski)" modunda eski V1 metni yazar."""
    at = _screen(store, monkeypatch, processed_df, "Bitcoin (BTC)", "BTC-USD")
    assert not at.exception
    assert any(t.startswith("Günlük test — kural: V1 + 200 günlük ortalama;") for t in _infos(at))
    assert not any(t.startswith("Günlük V1 testi") for t in _infos(at))
    at = next(r for r in at.radio if r.key == "signal_mode_choice").set_value(sr.V1).run()
    assert not at.exception
    assert any(t.startswith("Günlük V1 testi; kapanmış mum kararını") for t in _infos(at))
    assert not any(t.startswith("Günlük test — kural:") for t in _infos(at))


def test_ac04_panels_get_rule_decisions_and_candidates_keep_v1(store, monkeypatch, processed_df):
    """AC04 — Backtest sonrası perf_report_source["decisions"] kural kararları, ["v1_decisions"] V1 kararlarıdır; aday değerlendirmesi V1 kararlarını kullanır (kritik akış smoke)."""
    from types import SimpleNamespace

    import candidate_ui
    from app_helpers import click

    v1 = _v1(processed_df)
    seen = []
    monkeypatch.setattr(candidate_ui, "evaluate_candidates",
                        lambda symbol, frame, decisions, **k: seen.append(decisions) or SimpleNamespace(ok=True))
    at = click(_screen(store, monkeypatch, processed_df, "Bitcoin (BTC)", "BTC-USD", v1), "🚀 Backtest Başlat")
    assert not at.exception
    source = at.session_state["perf_report_source"]
    expected, _ = sr.rule_decisions("BTC-USD", processed_df, v1)
    assert source["decisions"] == expected != v1 and source["v1_decisions"] == v1
    assert any(s.value == "📑 Güvenilir Performans Raporu" for s in at.subheader)
    at = click(at, "Adayları değerlendir")
    assert not at.exception and seen and seen[-1] == v1


def test_ac02_hold_only_uses_the_same_period_as_the_hold_comparison(processed_df):
    """AC02 — Hissede gösterilen al-tut sonucu, al-tut karşılaştırmasıyla aynı günde başlar (ilk V1 kararından sonraki açılış)."""
    import performance_ui

    v1 = _v1(processed_df)
    alone = performance_ui.hold_only(processed_df, Decimal("10000"), v1)
    paired = performance_ui.hold_comparison(processed_df, {"total_return": Decimal("0")}, 10000, 1000, v1)
    assert alone.start == paired.start == processed_df.index[WARM_UP + 1].date()
    assert alone.price_change_pct == paired.price_change_pct
    assert alone.final_equity == Decimal("10000") * (1 + alone.price_change_pct / 100)
    assert performance_ui.hold_only(processed_df.iloc[:1], Decimal("10000"), v1) is None
