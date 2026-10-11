"""Spec 0018 — AL/SAT kararlarına ML girmez: kararlar kayan veri penceresinde sabit kalır."""
import signal_engine
from config import DecisionEngineConfig
from technical_analysis import build_v1_decisions
import pytest


@pytest.fixture(autouse=True)
def _fresh_caches():
    signal_engine._stable_cache.clear(); signal_engine._bar_score_cache.clear()
    yield
    signal_engine._stable_cache.clear(); signal_engine._bar_score_cache.clear()


def test_ac01_ml_is_off_for_decisions_by_default():
    """AC01 — Varsayılan ayar ML'yi AL/SAT kararından çıkarır."""
    assert DecisionEngineConfig.V1_INCLUDE_ML is False


def test_ac02_default_signal_and_backtest_never_train_ml(processed_df, monkeypatch):
    """AC02 — Ekran sinyali ve geçmiş test kararları varsayılan çağrıda ML modelini hiç çalıştırmaz."""
    calls = []
    monkeypatch.setattr(signal_engine, "_score_ml", lambda df: calls.append(1) or (0, [], True))
    signal_engine.generate_stable_signal(processed_df, "1d", data_is_closed=True)
    signal_engine.generate_composite_signal(processed_df, "1d")
    build_v1_decisions(processed_df.iloc[:230])
    assert calls == []


def test_ac03_decisions_do_not_change_when_the_window_slides(processed_df):
    """AC03 — Veri penceresi bir gün kayınca ortak günlerin kararları değişmez (deneyde ML'li 2–7 fark vardı)."""
    full = build_v1_decisions(processed_df)
    shifted = build_v1_decisions(processed_df.iloc[1:])
    common = sorted(set(full) & set(shifted))
    assert common and all(full[d] == shifted[d] for d in common)


def test_ac04_backtest_and_screen_agree_without_ml(processed_df):
    """AC04 — Geçmiş testin her bar kararı, ekranın o bardaki varsayılan (ML'siz) kararıyla aynıdır."""
    frame = processed_df.iloc[:240]
    decisions = build_v1_decisions(frame)
    for index in range(225, 240):
        screen = signal_engine.generate_stable_signal(frame.iloc[:index + 1], "1d", data_is_closed=True,
                                                      strict_components=True)
        label = "AL" if "AL" in screen.verdict else ("SAT" if "SAT" in screen.verdict else "BEKLE")
        if screen.unavailable_components:
            label = "BILESEN_YOK"
        assert decisions.get(frame.index[index], "BEKLE") == label, index
