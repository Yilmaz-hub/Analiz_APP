from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from market_validation import MarketPolicy, combine_source_history, normalize_pair, policy_for_symbol, validate_market_data


UTC = timezone.utc
NOW = datetime(2026, 9, 10, 0, 5, tzinfo=UTC)


def _bars(count=200, end="2026-09-09", source="YAHOO"):
    idx = pd.date_range(end=end, periods=count, freq="D", tz="UTC")
    return pd.DataFrame({
        "Open": [100.0] * count, "High": [102.0] * count,
        "Low": [99.0] * count, "Close": [101.0] * count,
        "Volume": [1000.0] * count, "Source": [source] * count,
    }, index=idx)


def _crypto_policy(**changes):
    values = dict(market="CRYPTO", expected_close=datetime(2026, 9, 10, tzinfo=UTC),
                  publication_delay=timedelta(minutes=5), required_bars=200)
    values.update(changes)
    return MarketPolicy(**values)


def test_ac16_open_candle_mutation_immunity():
    a, b = _bars(), _bars()
    a.loc[NOW, ["Open", "High", "Low", "Close", "Volume", "Source"]] = [101, 120, 80, 110, 10, "YAHOO"]
    b.loc[NOW, ["Open", "High", "Low", "Close", "Volume", "Source"]] = [101, 140, 60, 90, 20, "YAHOO"]
    ra = validate_market_data(a, NOW, _crypto_policy(), provider_open={NOW})
    rb = validate_market_data(b, NOW, _crypto_policy(), provider_open={NOW})
    pd.testing.assert_frame_equal(ra.usable, rb.usable)


def test_ac17_exact_close_validation_boundary():
    result = validate_market_data(_bars(end="2026-09-10"), NOW, _crypto_policy())
    assert result.usable.index[-1] == pd.Timestamp("2026-09-10", tz="UTC")


def test_ac18_keep_last_closed_candle():
    result = validate_market_data(_bars(), NOW, _crypto_policy(expected_close=datetime(2026, 9, 9, tzinfo=UTC)))
    assert len(result.usable) == 200


def test_ac20_history_n_minus_one():
    assert validate_market_data(_bars(199), NOW, _crypto_policy()).reason == "YETERSIZ_GECMIS"


def test_ac60_closed_session_not_stale():
    policy = MarketPolicy("BIST", datetime(2026, 9, 9, tzinfo=UTC), timedelta(minutes=30), 200)
    assert validate_market_data(_bars(), NOW, policy).is_valid


def test_ac61_open_tail_valid_history():
    frame = _bars()
    frame.loc[NOW] = [101, 102, 99, 101, 1000, "YAHOO"]
    assert validate_market_data(frame, NOW, _crypto_policy(), provider_open={NOW}).is_valid


def test_ac62_unready_required_fields():
    result = validate_market_data(_bars(), NOW, _crypto_policy(), components_ready=False)
    assert (result.is_valid, result.reason) == (False, "BILESEN_HAZIR_DEGIL")


def test_ac63_high_below_low():
    frame = _bars(); frame.iloc[-1, frame.columns.get_loc("High")] = 98
    assert validate_market_data(frame, NOW, _crypto_policy()).reason == "GECERSIZ_OHLC"


def test_ac104_usd_usdt_equivalence():
    assert normalize_pair("ETH/USD") == normalize_pair("ETH/USDT") == "ETH/USD*"


def test_ac107_source_fallback():
    yahoo, okx = _bars(source="YAHOO"), _bars(source="OKX")
    result = combine_source_history(okx, yahoo, primary_failed=True)
    assert result.fallback_used and set(result.usable["Source"]) == {"YAHOO"}


def test_ac116_crypto_delay():
    missing = _bars(end="2026-09-09")
    waiting = validate_market_data(missing, NOW - timedelta(seconds=1), _crypto_policy())
    stale = validate_market_data(missing, NOW, _crypto_policy())
    assert (waiting.status, stale.status) == ("YENI_VERI_BEKLENIYOR", "ESKI_VERI")
    assert not waiting.allow_new_action and not stale.allow_new_action


def test_ac117_yahoo_delay():
    close = datetime(2026, 9, 10, 20, 0, tzinfo=UTC)
    policy = MarketPolicy("YAHOO", close, timedelta(minutes=30), 200)
    missing = _bars(end="2026-09-09")
    assert validate_market_data(missing, close + timedelta(minutes=29, seconds=59), policy).status == "YENI_VERI_BEKLENIYOR"
    assert validate_market_data(missing, close + timedelta(minutes=30), policy).status == "ESKI_VERI"


def test_ac118_history_boundary():
    assert not validate_market_data(_bars(199), NOW, _crypto_policy()).is_valid
    assert validate_market_data(_bars(200), NOW, _crypto_policy(expected_close=datetime(2026, 9, 9, tzinfo=UTC))).is_valid


def test_ac119_open_flag():
    frame = _bars(end="2026-09-10")
    result = validate_market_data(frame, NOW + timedelta(days=1), _crypto_policy(), provider_open={frame.index[-1]})
    assert frame.index[-1] not in result.usable.index


@pytest.mark.parametrize("value", [0, -1])
def test_requirement_nonpositive_price(value):
    frame = _bars(); frame.iloc[-1, frame.columns.get_loc("Close")] = value
    assert validate_market_data(frame, NOW, _crypto_policy()).reason == "GECERSIZ_OHLC"


def test_requirement_reorders_and_deduplicates_equal_rows():
    frame = pd.concat([_bars(), _bars().iloc[[-1]]]).sort_index(ascending=False)
    result = validate_market_data(frame, NOW, _crypto_policy(expected_close=datetime(2026, 9, 9, tzinfo=UTC)))
    assert result.is_valid and result.usable.index.is_monotonic_increasing and len(result.usable) == 200


def test_requirement_rejects_conflicting_duplicate_rows():
    frame = pd.concat([_bars(), _bars().iloc[[-1]].assign(Close=77)])
    assert validate_market_data(frame, NOW, _crypto_policy()).reason == "CELISKILI_TEKRAR"


def test_crypto_policy_uses_utc_close_and_five_minute_delay():
    policy = policy_for_symbol("ETH-USD", datetime(2026, 9, 10, 12, tzinfo=UTC))
    assert policy.expected_close == datetime(2026, 9, 10, tzinfo=UTC)
    assert policy.expected_bar_date.isoformat() == "2026-09-09"
    assert policy.publication_delay == timedelta(minutes=5)


def test_crypto_policy_accepts_latest_completed_utc_daily_bar():
    policy = policy_for_symbol("ETH-USD", datetime(2026, 9, 10, 12, tzinfo=UTC))
    assert validate_market_data(_bars(end="2026-09-09"), NOW, policy).is_valid


def test_policy_removes_rows_after_latest_expected_closed_bar():
    frame = _bars(end="2026-09-09")
    frame.loc[pd.Timestamp("2026-09-10", tz="UTC")] = [101, 103, 98, 102, 1000, "OKX"]
    policy = policy_for_symbol("ETH-USD", datetime(2026, 9, 10, 12, tzinfo=UTC))

    result = validate_market_data(frame, datetime(2026, 9, 10, 12, tzinfo=UTC), policy)

    assert result.is_valid
    assert result.usable.index[-1] == pd.Timestamp("2026-09-09", tz="UTC")


def test_bist_policy_skips_weekend():
    policy = policy_for_symbol("THYAO.IS", datetime(2026, 9, 13, 12, tzinfo=UTC))
    assert policy.expected_close.date().isoformat() == "2026-09-11"


def test_us_close_policy_respects_daylight_saving_time():
    summer = policy_for_symbol("AAPL", datetime(2026, 9, 10, 23, tzinfo=UTC))
    winter = policy_for_symbol("AAPL", datetime(2026, 1, 15, 23, tzinfo=UTC))

    assert summer.expected_close.hour == 20
    assert winter.expected_close.hour == 21
