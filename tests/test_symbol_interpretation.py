"""Spec 0006 B — tire olmadan yazılan kripto sembollerinin her yerde aynı yorumu (R10, Q07, Q08)."""
from datetime import datetime, timezone
from decimal import Decimal

import pytest

import assets
import market_map
import market_validation

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("symbol", ["LINKUSD", "HBARUSD", "linkusd", " HBARUSD "])
def test_ac24_dashless_crypto_is_recognised_as_crypto_usd(symbol):
    """AC24 — `LINKUSD` ve `HBARUSD` kripto piyasası ve USD para birimi olarak tanınır."""
    assert market_map.market_of(symbol) == ("KRIPTO", "USD")


def test_ac24_dashless_usdt_pair_is_crypto_usdt():
    """AC24 — Tiresiz `LINKUSDT` kripto ve USDT para birimi olarak tanınır."""
    assert market_map.market_of("LINKUSDT") == ("KRIPTO", "USDT")


@pytest.mark.parametrize("symbol", ["LINKUSD", "HBARUSD", "LINK-USD"])
def test_ac24_dashless_crypto_gets_the_crypto_daily_close_rule(symbol):
    """AC24 — Tiresiz kripto, tireli kriptoyla aynı günlük kapanış kuralını (00:00 UTC) alır; ABD hissesi kuralı değil."""
    policy = market_validation.policy_for_symbol(symbol, NOW)
    assert policy.market == "CRYPTO"
    assert policy.expected_close == datetime(2026, 10, 4, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("symbol,expected", [
    ("AAPL", ("ABD", "USD")), ("THYAO.IS", ("BIST", "TRY")), ("XAU_GOLD", ("ALTIN", "USD")),
    ("GRAM_TRY", ("ALTIN", "TRY")), ("EURUSD=X", None), ("BTC-USD", ("KRIPTO", "USD")),
])
def test_ac25_non_crypto_symbols_keep_their_interpretation(symbol, expected):
    """AC25 — `AAPL`, `THYAO.IS`, `EURUSD=X`, `XAU_GOLD`, `GRAM_TRY` ve tireli kripto yorumları değişmez."""
    assert market_map.market_of(symbol) == expected


def test_ac25_policy_for_non_crypto_symbols_is_unchanged():
    """AC25 — Hisse, BIST ve altın için günlük kapanış politikası değişmez."""
    assert market_validation.policy_for_symbol("AAPL", NOW).market == "YAHOO"
    assert market_validation.policy_for_symbol("THYAO.IS", NOW).market == "BIST"
    assert market_validation.policy_for_symbol("XAU_GOLD", NOW).market == "GOLD_FUTURES"


def test_ac26_stored_symbol_is_not_rewritten(store):
    """AC26 — Depoda `LINKUSD` olarak kayıtlı sembol, yorumlama sonrası da `LINKUSD` olarak kalır."""
    store.write_doc(store.ASSETS_KEY, {"Chainlink": "LINKUSD"})
    assert market_map.market_of(assets.load_assets()["Chainlink"]) == ("KRIPTO", "USD")
    assert store.read_doc(store.ASSETS_KEY) == {"Chainlink": "LINKUSD"}
    ok, _, updated = assets.add_asset(assets.load_assets(), "Hedera", "HBARUSD")
    assert ok and updated["Hedera"] == "HBARUSD" and updated["Chainlink"] == "LINKUSD"


def test_ac28_add_asset_message_shows_how_the_symbol_will_be_read():
    """AC28 — Varlık eklerken `LINKUSD` için "LINK-USD, kripto olarak okunacak" bilgisi görünür."""
    ok, message, _ = assets.add_asset({}, "Chainlink", "LINKUSD")
    assert ok and "LINK-USD, kripto olarak okunacak" in message
    _, plain, _ = assets.add_asset({}, "Bitcoin", "BTC-USD")
    assert "okunacak" not in plain


def test_ac29_report_and_candidates_accept_dashless_crypto(trending_df):
    """AC29 — `LINKUSD` için performans raporu ve aday değerlendirmesi varlığı atlamaz."""
    from datetime import timedelta

    import candidate_ui
    import performance_ui

    decisions = {trending_df.index[260]: "AL", trending_df.index[280]: "SAT"}
    outcome = candidate_ui.evaluate_candidates(
        "LINKUSD", trending_df, decisions, notional=Decimal("1000"), capital=Decimal("10000"),
        quantity_step=Decimal("0.001"), costs=None, now=NOW)
    assert outcome.ok, outcome.reason
    backtest = {"daily_equity": [{"date": trending_df.index[0], "equity": "10000"},
                                 {"date": trending_df.index[1], "equity": "10100"}],
                "trades": [], "position": None, "net_verified": True}
    report = performance_ui.report_from_backtest("LINKUSD", backtest)
    assert report.status == "OK" and "USD" in report.report.results


def test_ac29_forward_runner_tracks_dashless_crypto(tmp_path):
    """AC29 — İleri takip koşucusu `HBARUSD` varlığını atlamaz ve karar kaydeder."""
    from datetime import timedelta

    import forward_runner
    import forward_tracker as ft
    from conftest import make_ohlcv
    from data_fetchers import process_data

    raw = make_ohlcv()
    raw.index = raw.index.tz_localize("UTC")
    frame, _ = process_data(raw, "fixture")
    last = frame.index[-1].date()
    report = forward_runner.run(ft.available_at("KRIPTO", last) + timedelta(minutes=5),
                                real_clock=False, fetch=lambda symbol: (frame, "fixture"),
                                include_ml=False, symbols=["HBARUSD"])
    assert report.recorded and not any("tanınmıyor" in note for note in report.notes)
