"""Spec 0005 Adım 5 / S3 — rejim sınıflandırıcısı REJIM-1 (Q02 karar kaydı)."""
from decimal import Decimal

import numpy as np
import pandas as pd

from regime_classifier import (
    DUSEN, REGIME_VERSION, YATAY, YUKSELEN, classify, classify_frame,
)


def test_ac85_classification_carries_classifier_version():
    """AC85 — Her piyasa koşulu gözlemi kendisini üreten sınıflandırıcının sürümüyle gelir."""
    label, version = classify(close=Decimal("110"), ma=Decimal("100"),
                              ma_before=Decimal("95"), score=60)
    assert (label, version) == (YUKSELEN, REGIME_VERSION) == (YUKSELEN, "REJIM-1")


def test_ac85_rising_needs_price_above_rising_ma_and_tradeable_score():
    """AC85 — Yükselen: kapanış MA üstünde, MA son 20 günde artmış ve puan ≥ 35; 34 puan yataydır."""
    assert classify(Decimal("110"), Decimal("100"), Decimal("95"), 35)[0] == YUKSELEN
    assert classify(Decimal("110"), Decimal("100"), Decimal("95"), 34)[0] == YATAY
    assert classify(Decimal("110"), Decimal("100"), Decimal("100"), 80)[0] == YATAY


def test_ac85_falling_needs_price_below_falling_ma():
    """AC85 — Düşen: kapanış MA altında ve MA son 20 günde azalmış; aksi halde yatay."""
    assert classify(Decimal("90"), Decimal("100"), Decimal("105"), 10)[0] == DUSEN
    assert classify(Decimal("90"), Decimal("100"), Decimal("95"), 10)[0] == YATAY


def test_ac85_unknown_inputs_are_not_classified():
    """AC85 — Yetersiz veri (MA veya puan yok) sınıf uydurmaz: sınıf None, sürüm yine yazılır."""
    assert classify(Decimal("90"), None, None, None) == (None, REGIME_VERSION)


def _trend_frame(n=200, slope=1.0):
    index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + slope * np.arange(n, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "ADX": 30.0}, index=index)


def test_ac85_frame_classification_is_point_in_time():
    """AC85 — Bir günün sınıfı sonraki fiyatlar değişince değişmez (gelecek verisi yok, AC12)."""
    frame = _trend_frame()
    before = classify_frame(frame)
    changed = frame.copy()
    changed.iloc[150:, changed.columns.get_loc("Close")] = 1.0
    after = classify_frame(changed)
    assert before.iloc[:150].equals(after.iloc[:150])
    assert before.iloc[149] == YUKSELEN
    assert before.iloc[0] is None  # ortalama oluşmadan sınıf yok
