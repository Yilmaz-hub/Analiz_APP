"""Portföy geçmiş testi paneli (spec 0019, P1).

Hesap `portfolio_backtest.py`'dedir; burası girdileri toplar ve sonucu gösterir. Kararlar varlığın yöntem
adayıdır (spec 0020): altcoinlerde kanal kırılımı 50/20, hisselerde (ve istenirse altında) stopsuz al-tut
(spec 0021), diğerlerinde trend adayı (V1 AL + kapanış > 200 günlük ortalama; çıkış V1). Tek kasa dolardır;
TL varlıklar günlük kurla çevrilir. Gösterim yuvarlaması yalnız burada.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import portfolio_backtest as pb

DATA_NOTE = ("Veri: seçilen dönem (2018'den uzun geçmiş: kripto Binance, altın/hisse/kur Yahoo; ya da uygulamanın "
             "son ~1000 günlük verisi); ilk ~200 gün göstergelerin oluşması için kullanılır, bu sürede işlem "
             "yapılmaz. Test BTC/ETH'nin ilk işlem gününden başlar; verisi daha geç "
             "başlayan varlığın payı o güne kadar nakit sayılır, al-tut da o gün alır. Sonuç geçmiş veridir, "
             "gelecekteki kazanç olasılığı değildir.")


@dataclass(frozen=True)
class PortfolioView:
    metrics: list          # [(etiket, strateji, al-tut)]
    rows: list             # varlık başına tablo
    notes: list


def _pct(value) -> str:
    return "hesaplanamıyor" if value is None else f"%{Decimal(value):.1f}"


METHOD_TEXT = {"KANAL": "Kanal 50/20", "TREND_SMA": "V1 + 200 gün ort.", "TUT": "Al-tut (stopsuz)"}
FX_SYMBOL = "USDTRY=X"


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
        hold = methods is not None and methods.get(symbol) == "TUT"
        rows.append({
            **row, "Ağırlık": f"%{Decimal(weights[symbol]):g}",
            "İşlem": "—" if hold else len(trades), "Kazanan": "—" if hold else sum(1 for t in trades if t.pnl > 0),
            "Net K/Z": f"{result.contribution.get(symbol, Decimal('0')):,.2f} {currency}".strip(),
        })
    used = set(methods.values()) if methods is not None else {"TREND_SMA"}
    notes = [f"Toplam {len(result.trades)} kapanmış işlem, {wins} kazanan.",
             *(["Kural: AL sinyali ve fiyat 200 günlük ortalamanın üstündeyse, o günkü toplam varlığın varlık "
                "ağırlığı kadar alım; çıkış SAT ya da 2,5 ATR stop. Nakit yetmezse giriş küçülür ya da atlanır."]
               if "TREND_SMA" in used else []),
             *(["Kanal 50/20: kapanış önceki 50 günün en yükseğini aşınca alım, önceki 20 günün en düşüğünün "
                "altına inince satış (2,5 ATR stop aynı)."] if "KANAL" in used else []),
             *(["Al-tut (stopsuz): ilk gün toplam varlığın ağırlığı kadar alınır, dönem sonuna kadar tutulur; "
                "Net K/Z dönem sonu değerine göredir."] if "TUT" in used else []),
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

    import history
    import market_map

    st.caption("Seçtiğiniz varlıklar tek kasayı paylaşır: bir varlık alımdayken kalan nakit diğer "
               "varlıklardaki fırsatlar için kullanılır.")
    defaults = [name for name, symbol in coin_map.items()
                if (market_map.market_of(symbol) or ("", ""))[0] == "KRIPTO"
                or market_map.market_of(symbol) == ("ALTIN", "USD")]
    chosen = st.multiselect("Varlıklar", list(coin_map), default=defaults, key="pf_assets")
    currencies = sorted({info[1] for info in (market_map.market_of(coin_map[name]) for name in chosen) if info})
    unsupported = [c for c in currencies if c not in pb.POOL_CURRENCIES]
    if unsupported:
        st.warning("Bu para birimleri dolara çevrilemiyor (" + ", ".join(unsupported) + "); bu varlıkları "
                   "çıkarın. Dolar ve TL varlıklar birlikte test edilebilir.")
    elif "TRY" in currencies:
        st.caption("TL varlıklar günlük USD/TRY kuruyla dolara çevrilir; sonuç dolar bazındadır.")
    with st.form("pf_settings_form"):
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
                                   "ortalamayla işlem yapar. Kapalı: hepsi V1 + 200 günlük ortalama. Hisseler her "
                                   "durumda al-tut (stopsuz) ile tutulur.")
        gold_hold = st.checkbox("Altını al-tut ile tut (stopsuz)", value=False, key="pf_gold_hold",
                                help="Kapalı: altın V1 + 200 günlük ortalamayla alınıp satılır; düştüğünde boşalan nakit "
                                     "kripto fırsatlarına gider (geçmiş testte portföyde daha iyi). Açık: altın ilk gün "
                                     "alınır ve tutulur.")
        period = st.selectbox("Test dönemi", list(history.PERIOD_TEXT), format_func=history.PERIOD_TEXT.get,
                              key="pf_period",
                              help="Uzun geçmişte ilk çalıştırma varlık başına 1–2 dakika sürebilir; kararlar "
                                   "saklanır, sonraki çalıştırmalar hızlıdır.")
        st.caption("Ağırlıklar toplamı en fazla %100 olmalıdır. Kalan pay boşta nakit olarak kasada kalır.")
        submitted = st.form_submit_button("📦 Portföy testini başlat", disabled=not chosen or bool(unsupported))

    if not submitted:
        return

    total_weight = sum((weights[n] for n in weights), Decimal("0"))
    if total_weight > Decimal("100"):
        st.error(f"Ağırlıklar toplamı %100'ü geçemez (girilen: %{total_weight:g}). Lütfen ağırlıkları toplamı en fazla %100 olacak şekilde ayarlayın.")
        return
    if total_weight <= Decimal("0"):
        st.warning("Ağırlıklar toplamı sıfırdan büyük olmalıdır. En az bir varlığa %0'dan büyük ağırlık verilmelidir.")
        return

    result, weights_by_symbol, methods = _run(chosen, weights, coin_map, source_pref, capital, commission, st,
                                              channel=channel, gold_hold=gold_hold, period=period)
    if result is None:
        return
    view = build_portfolio_view(result, weights_by_symbol, "USD", methods=methods)
    cash_reserve = Decimal("100") - total_weight
    st.info(f"Portföy ağırlık toplamı: %{total_weight:g}" +
            (f" | Boşta nakit: %{cash_reserve:g}" if cash_reserve > 0 else ""))
    table = pd.DataFrame([{"": label, "Strateji": s, "Al-tut (aynı ağırlıklar)": h} for label, s, h in view.metrics])
    st.dataframe(table, hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(view.rows), hide_index=True, width="stretch")
    _chart(st, result)
    for note in view.notes:
        st.caption(note)


def to_pool_frame(frame, fx):
    """TL mum tablosunu dolar kasasına hazırlar: fiyatlar kurla çevrilir, göstergeler (EMA, MACD, RSI, ATR…)
    dolar fiyatından yeniden hesaplanır. Yalnız fiyatı çevirmek, TL göstergeleri dolar fiyatıyla karıştırıp
    kararı bozar (spec 0021 QA). Hesaplanamazsa None."""
    from data_fetchers import process_data

    columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    converted = pb.to_usd(frame[columns], fx)
    if "Volume" not in converted.columns:
        converted["Volume"] = 0
    result, _ = process_data(converted, "USD/TRY ile çevrildi")
    return result


def _run(chosen, weights, coin_map, source_pref, capital, commission, st, channel=True, gold_hold=False,
         period="RECENT"):
    import history
    import strategy_candidates as sc
    from technical_analysis import build_v1_decisions
    from trade_execution import CostAssumptions
    from trade_settings import known_quantity_step
    from storage import StorageAccessError
    import market_map

    frames, decisions, steps, weights_by_symbol, missing, v1_days, methods = {}, {}, {}, {}, [], {}, {}
    ready, reference, crypto, notes = {}, [], [], []
    fx = None
    progress = st.progress(0.0)
    for n, name in enumerate(chosen):
        symbol = coin_map[name]
        loaded = history.load_history(source_pref, symbol, period)
        frame, info = loaded.frame, market_map.market_of(symbol)
        if loaded.note:
            notes.append(loaded.note)
        if frame is None or len(frame) < 260 or info is None:
            missing.append(name)
            continue
        if pb.POOL_CURRENCIES.get(info[1]) == "TRY":
            if fx is None:
                fx = history.load_history(source_pref, FX_SYMBOL, period).frame
            if fx is None or len(fx) == 0:
                missing.append(f"{name} (USD/TRY kuru alınamadı)")
                continue
            frame = to_pool_frame(frame, fx)
            if frame is None or len(frame) < 260:
                missing.append(f"{name} (dolar verisi yetersiz)")
                continue
        method = sc.asset_method(symbol, gold_hold=gold_hold)
        if not channel and method == "KANAL":
            method = "TREND_SMA"
        methods[name] = method
        if method == sc.TUT:
            decisions[name] = {}                    # al-tut: karar yok, V1 hesaplanmaz
            ready[name] = frame.index[0]            # ısınma gerekmez: ilk mumda alınabilir
        else:
            if period == history.LONG:
                v1 = history.cached_v1_decisions(symbol, frame, progress=getattr(st, "caption", None),
                                                  variant="USD" if pb.POOL_CURRENCIES.get(info[1]) == "TRY" else "")
            else:
                v1 = build_v1_decisions(frame)
            v1_days[name] = list(v1)
            first = pb.first_tradable_day(frame, v1_days[name])
            if first is None:
                missing.append(f"{name} (gösterge oluşmadı)")
                continue
            candidate = sc.channel_candidate(info[0]) if method == "KANAL" else sc.trend_candidate(info[0])
            # Spec 0025 R02: ısınma döneminde karar yok (V1'in ilk karar gününden önce işlem yapılmaz; 0023 ile aynı).
            decisions[name] = {day: verdict for day, verdict in sc.candidate_decisions(frame, v1, candidate, None).items()
                               if day >= min(v1)}
            ready[name] = first
        if info[0] == "KRIPTO":
            crypto.append(name)
            parts = market_map.crypto_parts(symbol)
            if parts is not None and parts[0] in ("BTC", "ETH"):
                reference.append(name)
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
    for note in notes:
        st.warning(note)
    if missing:
        st.warning("Verisi yetersiz ya da piyasası tanınmayan varlıklar teste alınmadı: " + ", ".join(missing))
    if not frames:
        st.warning("Test edilecek varlık kalmadı.")
        return None, {}, {}
    start, late = pb.portfolio_start(ready, reference or crypto)
    for name, day in sorted(late.items(), key=lambda item: item[1]):
        st.warning(f"{name}: {day.date()} öncesinde verisi ya da göstergesi olmadığı için işlem yapılamadı; "
                   f"o tarihe kadar payı (%{weights_by_symbol[name]:g}) nakit sayıldı.")
    result = pb.run_portfolio(frames, decisions, weights_by_symbol, capital=capital,
                              costs=CostAssumptions(None, None, commission), quantity_steps=steps,
                              start=start, hold={n for n, m in methods.items() if m == sc.TUT},
                              available_from=ready)
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
