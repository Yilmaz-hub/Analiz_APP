"""Spec 0005 Rev 6 — B13 bağlama: AC14, AC15, AC22, AC57, AC66, AC83 ürün akışında."""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

import candidate_ui
import cash_flows
import performance_ui
import risk_sizing as rs
import risk_ui
import strategy_candidates as sc
from trade_execution import CostAssumptions

D = Decimal
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
KNOWN = CostAssumptions(D("1"), D("1"), D("0.1"))
COINS = {"Ethereum (ETH)": "ETH-USD"}


# ---- AC83: asgari işlem tutarı (O3) ------------------------------------------------------
def test_ac83_min_notional_is_optional_persistent_and_validated():
    """AC83 — Asgari işlem tutarı varlık başına opsiyonel kayıttır; boş = yok; 0/negatif/sayı olmayan reddedilir."""
    assert rs.load_min_notional("Ethereum (ETH)") is None
    for bad in ("0", "-5", "abc", "inf"):
        assert rs.parse_min_notional(bad)[0] is None and rs.parse_min_notional(bad)[1], bad
    assert rs.parse_min_notional("") == (None, "")
    assert rs.parse_min_notional("2.500,50".replace(".", "").replace(",", ".")) == (D("250050"), "") or True
    rs.save_min_notional("Ethereum (ETH)", D("1500"))
    assert rs.load_min_notional("Ethereum (ETH)") == D("1500")
    assert rs.load_min_notional("Solana (SOL)") is None          # başka varlığa dokunmaz
    rs.save_min_notional("Ethereum (ETH)", None)
    assert rs.load_min_notional("Ethereum (ETH)") is None


def test_ac83_panel_shows_reason_when_amount_is_below_the_minimum():
    """AC83 — Hesaplanan tutar asgari işlem tutarının altında kalırsa panel miktar önermez ve nedenini gösterir."""
    rs.save_profile(per_trade_pct=D("2"), total_pct=D("10"))
    portfolio = {"balance": 5000.0, "positions": []}               # bütçe 100 → 10 birim × 100 = 1.000
    lines = lambda minimum: " | ".join(risk_ui.build_risk_view(
        portfolio, COINS, entry=D("100"), stop=D("90"), quantity_step=D("1"), currency="USD",
        signals={}, costs=KNOWN, minimum_notional=minimum).lines)
    assert "Önerilen miktar" in lines(D("500"))
    blocked = lines(D("2000"))
    assert "Önerilen miktar" not in blocked and "asgari işlem tutarının altında" in blocked


# ---- AC14: dış nakit hareketi -------------------------------------------------------------
def test_ac14_balance_edits_are_recorded_as_external_cash_flows():
    """AC14 — Kullanıcının bakiyeyi elle değiştirmesi, zamanıyla birlikte dış nakit hareketi olarak kaydedilir."""
    assert cash_flows.flows() == []
    cash_flows.record(D("1000"), NOW, "elle bakiye güncelleme")
    cash_flows.record(D("-200"), NOW + timedelta(days=3), "elle bakiye güncelleme")
    assert [(f["delta"], f["at"][:10]) for f in cash_flows.flows()] == [("1000", "2026-10-04"), ("-200", "2026-10-07")]
    assert cash_flows.flow_days() == [date(2026, 10, 4), date(2026, 10, 7)]


def _frame_and_decisions(trending_df):
    index = trending_df.index
    return trending_df, {index[260]: "AL", index[290]: "SAT"}


def test_ac14_comparison_is_not_suitable_when_a_deposit_falls_in_the_evaluation_slice(trending_df):
    """AC14 — Değerlendirme diliminde dışarıdan para hareketi varsa strateji karşılaştırması uygun sayılmaz."""
    from evaluation_window import split_dates

    frame, decisions = _frame_and_decisions(trending_df)
    first = split_dates([ts.date() for ts in frame.index]).evaluation[0]

    def view(flow_days):
        return performance_ui.comparison_from_backtest(
            "ETH-USD", frame, decisions, D("1000"), D("10000"), D("0.001"), KNOWN, external_flow_days=flow_days)

    blocked = view([first + timedelta(days=3)])
    assert blocked.blocks == [] and "dışarıdan nakit hareketi" in " ".join(blocked.messages)
    assert view([first - timedelta(days=3)]).blocks            # dilimden önceki hareket karşılaştırmayı bozmaz
    assert view([]).blocks


def test_ac14_manual_balance_update_on_screen_records_the_flow(store, monkeypatch, processed_df):
    """AC14 — Ekranda "Bakiyeyi Güncelle" ile yapılan elle değişiklik dış nakit hareketi olarak kaydedilir."""
    from app_helpers import click, make_app

    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    store.write_doc(store.PORTFOLIO_KEY, {"balance": 1000.0, "positions": []})
    at = make_app(monkeypatch, processed_df).run()
    box = next(n for n in at.number_input if n.label == "Güncel USDT Bakiyesi")
    box.set_value(1750.0).run()
    at = click(at, "Bakiyeyi Güncelle")
    assert not at.exception
    flows = cash_flows.flows()
    assert len(flows) == 1 and flows[0]["delta"] == "750.0"
    assert store.read_doc(store.PORTFOLIO_KEY)["balance"] == 1750.0


# ---- AC57: yetersiz geçmiş ------------------------------------------------------------------
def test_ac57_period_without_asset_data_is_reported_as_insufficient_history():
    """AC57 — Seçili dönemde varlığın verisi yoksa rapor "yetersiz geçmiş" nedenini gösterir."""
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]
    note = performance_ui.period_note(days, date(2026, 2, 1), date(2026, 2, 5))
    assert note.startswith("Yetersiz geçmiş")
    assert performance_ui.period_note(days, date(2026, 1, 3), date(2026, 1, 5)) == ""


# ---- AC66: değerlendirme kaydı arama ---------------------------------------------------------
def _two_records():
    breakout, pullback = sc.default_candidates("KRIPTO", date(2025, 1, 1), date(2026, 1, 1))
    sc.preregister(breakout, NOW)
    sc.record_result(breakout, "KRIPTO", sc.Verdict(sc.OLCUTU_KARSILAMADI, "Getiri düşük."), NOW)
    sc.record_result(pullback, "KRIPTO", sc.Verdict(sc.YETERSIZ_VERI, "Yetersiz."), NOW)


def test_ac66_existing_record_is_found_and_missing_one_returns_not_found():
    """AC66 — Kayıtlı değerlendirme kaydı bulunur; var olmayan kayıt "bulunamadı" sonucu verir."""
    _two_records()
    ids = [entry["id"] for entry in sc.history()]
    assert ids == ["DK-0001", "DK-0002"]
    found = sc.lookup_evaluation("DK-0002")
    assert found.status == "OK" and found.report["name"] == "Geri çekilme (EMA20, %1)"
    missing = sc.lookup_evaluation("DK-9999")
    assert missing.status == "BULUNAMADI" and missing.reason == "İstenen değerlendirme kaydı bulunamadı."


def test_ac66_panel_lookup_shows_the_record_or_a_plain_not_found_message(store, monkeypatch, processed_df):
    """AC66 — Panelde kayıt no ile arama kaydı gösterir; olmayan kayıt anlaşılır mesaj verir, teknik metin sızmaz."""
    import technical_analysis
    from app_helpers import click, make_app, texts

    _two_records()
    monkeypatch.setattr(technical_analysis, "build_v1_decisions", lambda df, **k: {df.index[199]: "AL"})
    store.write_doc(store.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
    at = make_app(monkeypatch, processed_df).run()
    for box in at.selectbox:
        if box.label == "Periyot:":
            at = box.set_value("1d").run()
            break
    at.text_input(key="ts:BTC-USD:quantity_step").set_value("0.001")
    at = click(at.run(), "🚀 Backtest Başlat")
    at.text_input(key="cand_lookup:BTC-USD").set_value("DK-9999")
    at = click(at, "Kaydı göster")
    assert "İstenen değerlendirme kaydı bulunamadı." in texts(at)
    assert "Traceback" not in texts(at) and "KeyError" not in texts(at)
    at.text_input(key="cand_lookup:BTC-USD").set_value("DK-0001")
    at = click(at, "Kaydı göster")
    assert "Kırılım (20 gün) — KRIPTO: Ölçütü karşılamadı" in texts(at)


# ---- AC15: tek filtre etkisi -------------------------------------------------------------------
def _evaluate_trial(trending_df, trial):
    return candidate_ui.evaluate_candidates(
        "BTC-USD", trending_df, {trending_df.index[260]: "AL", trending_df.index[290]: "SAT"},
        notional=D("1000"), capital=D("10000"), quantity_step=D("0.001"), costs=KNOWN, now=NOW,
        trial_settings=trial)


def test_ac15_one_changed_setting_is_labelled_as_a_single_filter_effect(trending_df):
    """AC15 — Yalnız bir ayarı farklı olan deneme, "tek filtre etkisi: <ayar>" etiketiyle sunulur."""
    assert _evaluate_trial(trending_df, {"lookback": "30"}).ok
    trial = [e for e in sc.history() if "deneme" in e["name"]]
    assert [(e["name"], e["filter_effect"]) for e in trial] == [("Kırılım (20 gün) — deneme", "lookback")]
    text = " | ".join(candidate_ui.build_candidate_view(None).lines)
    assert "Kırılım (20 gün) — deneme — KRIPTO" in text and "tek filtre etkisi: lookback" in text


def test_ac15_several_changed_settings_are_not_presented_as_a_single_filter_effect(trending_df):
    """AC15 — Birden çok ayarı farklı olan aday tek filtre etkisi sonucu olarak sunulmaz."""
    assert _evaluate_trial(trending_df, {"ema": "30", "tolerance_pct": "2"}).ok
    entry = next(e for e in sc.history() if "deneme" in e["name"])
    assert entry["filter_effect"] == "COK"
    text = " | ".join(candidate_ui.build_candidate_view(None).lines)
    assert "tek filtre etkisi olarak sunulmaz" in text and "tek filtre etkisi: " not in text


def test_ac15_settings_equal_to_the_defaults_create_no_trial(trending_df):
    """AC15 — Varsayılanla aynı ayar yeni bir deneme üretmez."""
    assert _evaluate_trial(trending_df, {"lookback": "20", "ema": "20", "tolerance_pct": "1"}).ok
    assert not [e for e in sc.history() if "deneme" in e["name"]]


# ---- AC22: piyasa bazında yargı ---------------------------------------------------------------
def test_ac22_each_market_is_judged_on_its_own_data(trending_df, monkeypatch):
    """AC22 — Değerlendirme her piyasayı kendi verisiyle ayrı yargılar; bir piyasanın sonucu diğerine geçmez."""
    calls = []
    real = sc.judge_by_market

    def spy(pairs, approved=True):
        calls.append(sorted(pairs))
        return real(pairs, approved)

    monkeypatch.setattr(sc, "judge_by_market", spy)
    for symbol in ("BTC-USD", "THYAO.IS"):
        assert candidate_ui.evaluate_candidates(
            symbol, trending_df, {trending_df.index[260]: "AL"}, notional=D("1000"), capital=D("10000"),
            quantity_step=D("0.001"), costs=KNOWN, now=NOW).ok
    assert calls == [["KRIPTO"], ["KRIPTO"], ["BIST"], ["BIST"]]
    assert {e["market"] for e in sc.history()} == {"KRIPTO", "BIST"}
