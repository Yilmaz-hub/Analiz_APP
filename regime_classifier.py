"""Piyasa koşulu sınıflandırıcısı REJIM-1 (spec 0005, Adım 5 / S3, Q02 karar kaydı).

Yeni eşik türetilmez; mevcut ayarlar kullanılır (`RegimeConfig`):

* **Yükselen:** kapanış `MA_PERIOD` günlük basit ortalamanın üstünde, ortalamanın
  son `SLOPE_LOOKBACK` gündeki değişimi pozitif ve rejim puanı ≥ `MIN_TRADEABLE_SCORE`.
* **Düşen:** kapanış ortalamanın altında ve ortalamanın aynı süredeki değişimi negatif.
* **Yatay:** diğer tüm günler.

Her sonuç sürüm etiketiyle döner (AC85). Bir günün sınıfı yalnız o güne kadarki
veriden hesaplanır (gelecek verisi yok). Eşik değişirse sürüm de değişmelidir.
"""
from __future__ import annotations

from decimal import Decimal

from config import RegimeConfig

REGIME_VERSION = "REJIM-1"
YUKSELEN = "YUKSELEN"
DUSEN = "DUSEN"
YATAY = "YATAY"


def classify(close: Decimal, ma: Decimal | None, ma_before: Decimal | None,
             score: int | None) -> tuple[str | None, str]:
    """`(sınıf, sürüm)`; ortalama veya puan yoksa sınıf `None` (uydurulmaz)."""
    if ma is None or ma_before is None or score is None:
        return None, REGIME_VERSION
    if close > ma and ma > ma_before and score >= RegimeConfig.MIN_TRADEABLE_SCORE:
        return YUKSELEN, REGIME_VERSION
    if close < ma and ma < ma_before:
        return DUSEN, REGIME_VERSION
    return YATAY, REGIME_VERSION


def _decimal(value) -> Decimal | None:
    if value is None or value != value:  # NaN
        return None
    return Decimal(str(value))


def classify_frame(frame):
    """Günlük mum tablosu → her gün için sınıf (pandas Series, sınıf yoksa None)."""
    import pandas as pd

    from technical_analysis import calculate_regime_score

    close = frame["Close"]
    ma = close.rolling(RegimeConfig.MA_PERIOD).mean()
    ma_before = ma.shift(RegimeConfig.SLOPE_LOOKBACK)
    labels = []
    for position in range(len(frame)):
        result = calculate_regime_score(frame.iloc[:position + 1])
        score = None if result is None else result["score"]
        labels.append(classify(_decimal(close.iloc[position]), _decimal(ma.iloc[position]),
                               _decimal(ma_before.iloc[position]), score)[0])
    return pd.Series(labels, index=frame.index, dtype=object)


def classify_latest(frame) -> tuple[str | None, str]:
    """Tablonun son günü için `(sınıf, sürüm)`; yalnız o güne kadarki veriyle."""
    from technical_analysis import calculate_regime_score

    close = frame["Close"]
    ma = close.rolling(RegimeConfig.MA_PERIOD).mean()
    ma_before = ma.shift(RegimeConfig.SLOPE_LOOKBACK)
    result = calculate_regime_score(frame)
    return classify(_decimal(close.iloc[-1]), _decimal(ma.iloc[-1]), _decimal(ma_before.iloc[-1]),
                    None if result is None else result["score"])
