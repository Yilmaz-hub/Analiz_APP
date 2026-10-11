"""Portföy geçmiş testi paneli (spec 0019, P1).

Hesap `portfolio_backtest.py`'dedir; burası girdileri toplar ve sonucu gösterir. Kararlar varlığın yöntem
adayıdır (spec 0020): altcoinlerde kanal kırılımı 50/20, diğerlerinde trend adayı (V1 AL + kapanış > 200 günlük
ortalama; çıkış V1). Gösterim yuvarlaması yalnız burada.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import portfolio_backtest as pb

DATA_NOTE = ("Veri, uygulamanın günlük verisidir (son ~1000 gün); ilk ~200 gün göstergelerin oluşması için "
             "kullanılır, strateji ve al-tut ilk karardan sonraki açılışta birlikte başlar. Sonuç geçmiş veridir, "
             "gelecekteki kazanç olasılığı değildir.")


@dataclass(frozen=True)
class PortfolioView:
    metrics: list          # [(etiket, strateji, al-tut)]
    rows: list             # varlık başına tablo
    notes: list


def _pct(value) -> str:
    return "hesaplanamıyor" if value is None else f"%{Decimal(value):.1f}"


METHOD_TEXT = {"KANAL": "Kanal 50/20", "TREND_SMA": "V1 + 200 gün ort."}


def build_portfolio_view(result: pb.PortfolioResult, weights: dict, currency: str = "",
                         methods: dict | None = None) -> PortfolioView:
    strategy, hold = pb.summary(result.equity), pb.summary(result.hold_equity)
    wins = sum(1 for t in result.trades if t.pnl > 0)
    metrics = [
        ("Yıllık getiri", _pct(strategy["annual"]), _pct(hold["annual"])),
        ("Toplam getiri", _pct(strategy["total"]), _pct(hold["total"])),
        ("En büyük düşüş", _pct(strategy["drawdown"]), _pct(hold["drawdown"])),
    ]
    rows = []
    for symbol in weights:
        trades = [t for t in result.trades if t.symbol == symbol]
        row = {"Varlık": symbol}
        if methods is not None:
            row["Yöntem"] = METHOD_TEXT.get(methods.get(symbol), methods.get(symbol) or "—")
        rows.append({
            **row, "Ağırlık": f"%{Decimal(weights[symbol]):g}",
            "İşlem": len(trades), "Kazanan": sum(1 for t in trades if t.pnl > 0),
            "Net K/Z": f"{result.contribution.get(symbol, Decimal('0')):,.2f} {currency}".strip(),
        })
    notes = [f"Toplam {len(result.trades)} kapanmış işlem, {wins} kazanan.",
             "Kural: AL sinyali ve fiyat 200 günlük ortalamanın üstündeyse, o günkü toplam varlığın varlık "
             "ağırlığı kadar alım; çıkış SAT ya da 2,5 ATR stop. Nakit yetmezse giriş küçülür ya da atlanır.",
             *(["Kanal 50/20: kapanış önceki 50 günün en yükseğini aşınca alım, önceki 20 günün en düşüğünün "
                "altına inince satış (2,5 ATR stop aynı)."] if methods and "KANAL" in methods.values() else []),
             DATA_NOTE]
    if result.start and strategy["years"]:
        notes.insert(0, f"Dönem: {result.start} – {result.end} ({Decimal(strategy['years']):.1f} yıl).")
    if result.approximate:
        notes.append("Miktar adımı girilmemiş varlıklarda kesirli miktar kullanıldı; sonuç yaklaşıktır: "
                     + ", ".join(result.approximate) + ".")
    return PortfolioView(metrics, rows, notes)


def render_portfolio_panel(coin_map: dict, source_pref: str) -> None:
    import pandas as pd
    import streamlit as st

    import market_map

    st.caption("Seçtiğiniz varlıklar tek kasayı paylaşır: bir varlık alımdayken kalan nakit diğer "
               "varlıklardaki fırsatlar için kullanılır.")
    defaults = [name for name, symbol in coin_map.items()
                if (market_map.market_of(symbol) or ("", ""))[0] == "KRIPTO"
                or market_map.market_of(symbol) == ("ALTIN", "USD")]
    chosen = st.multiselect("Varlıklar", list(coin_map), default=defaults, key="pf_assets")
    weights = {}
    if chosen:
        columns = st.columns(min(4, len(chosen)))
        for n, name in enumerate(chosen):
            default = float(pb.default_weight(coin_map[name]))
            weights[name] = Decimal(str(columns[n % len(columns)].number_input(
                f"{name} ağırlığı (%)", min_value=0.0, max_value=100.0, value=default, step=1.0,
                key=f"pf_w:{name}")))
    left, right = st.columns(2)
    capital = Decimal(str(left.number_input("Başlangıç sermayesi", min_value=100.0, value=10000.0,
                                            step=1000.0, key="pf_capital")))
    commission = Decimal(str(right.number_input("Komisyon (%)", min_value=0.0, max_value=5.0, value=0.1,
                                                step=0.05, key="pf_fee")))
    channel = st.checkbox("Altcoinlerde kanal kırılımı (50/20)", value=True, key="pf_channel",
                          help="Açık: BTC/ETH dışındaki kriptolar kanal kırılımıyla, diğerleri V1 + 200 günlük "
                               "ortalamayla işlem yapar. Kapalı: hepsi V1 + 200 günlük ortalama.")
    currencies = sorted({(market_map.market_of(coin_map[name]) or ("", "?"))[1] for name in chosen})
    mixed = len(currencies) > 1
    if mixed:
        st.warning("Seçilen varlıklar farklı para birimlerinde (" + ", ".join(currencies) + "); tek kasada "
                   "toplanamaz. Aynı para birimindeki varlıkları seçin.")
    if not st.button("📦 Portföy testini başlat", key="pf_run", disabled=not chosen or mixed):
        return
    result, weights_by_symbol, methods = _run(chosen, weights, coin_map, source_pref, capital, commission, st,
                                              channel=channel)
    if result is None:
        return
    view = build_portfolio_view(result, weights_by_symbol, methods=methods)
    table = pd.DataFrame([{"": label, "Strateji": s, "Al-tut (aynı ağırlıklar)": h} for label, s, h in view.metrics])
    st.dataframe(table, hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(view.rows), hide_index=True, width="stretch")
    _chart(st, result)
    for note in view.notes:
        st.caption(note)


def _run(chosen, weights, coin_map, source_pref, capital, commission, st, channel=True):
    import strategy_candidates as sc
    from data_fetchers import get_market_data
    from technical_analysis import build_v1_decisions
    from trade_execution import CostAssumptions
    from trade_settings import known_quantity_step
    from storage import StorageAccessError
    import market_map

    frames, decisions, steps, weights_by_symbol, missing, v1_days, methods = {}, {}, {}, {}, [], {}, {}
    progress = st.progress(0.0)
    for n, name in enumerate(chosen):
        symbol = coin_map[name]
        frame, _ = get_market_data(source_pref, symbol, "1d")
        info = market_map.market_of(symbol)
        if frame is None or len(frame) < 260 or info is None:
            missing.append(name)
            continue
        v1 = build_v1_decisions(frame)
        v1_days[name] = list(v1)
        candidate = sc.method_candidate(symbol) if channel else sc.trend_candidate(info[0])
        decisions[name] = sc.candidate_decisions(frame, v1, candidate, None)
        methods[name] = candidate.entry_rule
        frames[name] = frame
        weights_by_symbol[name] = weights[name]
        try:
            step = known_quantity_step(name)
        except StorageAccessError:
            step = None
        if step is not None:
            steps[name] = step
        progress.progress((n + 1) / len(chosen))
    progress.empty()
    if missing:
        st.warning("Verisi yetersiz ya da piyasası tanınmayan varlıklar teste alınmadı: " + ", ".join(missing))
    if not frames:
        st.warning("Test edilecek varlık kalmadı.")
        return None, {}, {}
    starts = [day for day in (pb.first_tradable_day(frames[n], v1_days[n]) for n in frames) if day is not None]
    result = pb.run_portfolio(frames, decisions, weights_by_symbol, capital=capital,
                              costs=CostAssumptions(None, None, commission), quantity_steps=steps,
                              start=min(starts) if starts else None)
    return result, weights_by_symbol, methods


def _chart(st, result):
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[d for d, _ in result.equity], y=[float(v) for _, v in result.equity],
                                name="Strateji", mode="lines"))
    figure.add_trace(go.Scatter(x=[d for d, _ in result.hold_equity], y=[float(v) for _, v in result.hold_equity],
                                name="Al-tut", mode="lines", line=dict(dash="dot")))
    figure.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
