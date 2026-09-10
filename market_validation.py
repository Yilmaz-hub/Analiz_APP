from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import math
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class MarketPolicy:
    market: str
    expected_close: datetime
    publication_delay: timedelta
    required_bars: int = 200
    expected_bar_date: object | None = None


@dataclass(frozen=True)
class MarketValidation:
    usable: pd.DataFrame
    is_valid: bool = False
    reason: str = ""
    status: str = ""
    allow_new_action: bool = False
    fallback_used: bool = False


def _previous_weekday(day):
    candidate = day
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def policy_for_symbol(symbol, now):
    """Return the approved daily close policy for a supported V1 symbol."""
    normalized = symbol.upper()
    utc_now = now.astimezone(timezone.utc)
    if normalized.endswith(("-USD", "/USD", "-USDT", "/USDT")):
        close = datetime.combine(utc_now.date(), time.min, tzinfo=timezone.utc)
        return MarketPolicy(
            "CRYPTO", close, timedelta(minutes=5), 200,
            expected_bar_date=(close - timedelta(days=1)).date(),
        )

    if normalized.endswith(".IS"):
        close_time = time(18, 10)
        close_zone = ZoneInfo("Europe/Istanbul")
        market = "BIST"
    elif normalized in {"XAU_GOLD", "GC=F"}:
        close_time = time(17, 0)
        close_zone = ZoneInfo("America/New_York")
        market = "GOLD_FUTURES"
    else:
        close_time = time(16, 0)
        close_zone = ZoneInfo("America/New_York")
        market = "YAHOO"

    session_day = _previous_weekday(utc_now.date())
    candidate = datetime.combine(session_day, close_time, tzinfo=close_zone).astimezone(timezone.utc)
    if session_day == utc_now.date() and utc_now < candidate:
        session_day = _previous_weekday(session_day - timedelta(days=1))
        candidate = datetime.combine(session_day, close_time, tzinfo=close_zone).astimezone(timezone.utc)
    return MarketPolicy(
        market, candidate, timedelta(minutes=30), 200,
        expected_bar_date=session_day,
    )


def validate_market_data(frame, now, policy, **kwargs):
    provider_open = {pd.Timestamp(value) for value in kwargs.get("provider_open", set())}
    components_ready = kwargs.get("components_ready", True)
    if frame is None or frame.empty:
        return MarketValidation(pd.DataFrame(), reason="VERI_YOK", status="VERI_YOK")

    data = frame.copy().sort_index()
    duplicate_times = data.index[data.index.duplicated(keep=False)].unique()
    for stamp in duplicate_times:
        rows = data.loc[[stamp]]
        if not rows.eq(rows.iloc[0]).all(axis=None):
            return MarketValidation(pd.DataFrame(), reason="CELISKILI_TEKRAR", status="GECERSIZ_VERI")
    data = data[~data.index.duplicated(keep="first")]
    data = data.loc[~data.index.isin(provider_open)]
    if policy.expected_bar_date is not None:
        data = data.loc[[stamp.date() <= policy.expected_bar_date for stamp in data.index]]

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(name not in data.columns for name in required):
        return MarketValidation(data, reason="ZORUNLU_ALAN_EKSIK", status="GECERSIZ_VERI")
    numeric = data[required].apply(pd.to_numeric, errors="coerce")
    finite = numeric.map(lambda value: math.isfinite(float(value))).all(axis=None)
    prices = numeric[["Open", "High", "Low", "Close"]]
    coherent = (
        finite
        and prices.gt(0).all(axis=None)
        and numeric["Volume"].ge(0).all()
        and numeric["High"].ge(prices[["Open", "Low", "Close"]].max(axis=1)).all()
        and numeric["Low"].le(prices[["Open", "High", "Close"]].min(axis=1)).all()
    )
    if not coherent:
        return MarketValidation(data, reason="GECERSIZ_OHLC", status="GECERSIZ_VERI")
    if len(data) < policy.required_bars:
        return MarketValidation(data, reason="YETERSIZ_GECMIS", status="YETERSIZ_GECMIS")
    if not components_ready:
        return MarketValidation(data, reason="BILESEN_HAZIR_DEGIL", status="BILESEN_HAZIR_DEGIL")

    expected_day = policy.expected_bar_date or pd.Timestamp(policy.expected_close).date()
    latest_day = data.index[-1].date()
    explicit_current_open = any(stamp.date() >= expected_day for stamp in provider_open)
    if latest_day < expected_day and not explicit_current_open:
        boundary = policy.expected_close + policy.publication_delay
        status = "YENI_VERI_BEKLENIYOR" if now < boundary else "ESKI_VERI"
        return MarketValidation(data, reason=status, status=status, allow_new_action=False)
    return MarketValidation(data, True, "", "GECERLI", True)


def normalize_pair(symbol):
    normalized = symbol.strip().upper().replace("-", "/")
    if normalized.endswith("/USDT"):
        normalized = normalized[:-5] + "/USD"
    return normalized + "*" if normalized.endswith("/USD") else normalized


def combine_source_history(primary, fallback, primary_failed=False):
    selected = fallback.copy() if primary_failed else primary.copy()
    if "Source" in selected and selected["Source"].nunique() > 1:
        return MarketValidation(pd.DataFrame(), reason="KARISIK_KAYNAK", status="GECERSIZ_VERI")
    return MarketValidation(selected, True, "", "GECERLI", True, primary_failed)
