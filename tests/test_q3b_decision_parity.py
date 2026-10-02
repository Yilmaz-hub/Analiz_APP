"""QA Y3 (R06): geçmiş testin bar kararı, ekranın/sanal takibin aynı veride vereceği karardır."""
import pytest

import ml_models
import signal_engine
from signal_engine import generate_stable_signal
from technical_analysis import build_v1_decisions


def _label(verdict):
    return "AL" if "AL" in verdict else ("SAT" if "SAT" in verdict else "BEKLE")


def _fake_ml(monkeypatch):
    """Kararı etkileyecek, yalnız veriye bağlı (deterministik) bir ML sonucu."""
    def fake(df):
        phase = len(df) % 4
        direction = ("BULLISH", "BEARISH", "NEUTRAL", "BULLISH")[phase]
        return {"direction": direction, "confidence": 80.0, "predicted_change_pct": 1.0}
    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", fake)


@pytest.fixture(autouse=True)
def _fresh_caches():
    signal_engine._stable_cache.clear(); signal_engine._bar_score_cache.clear()
    yield
    signal_engine._stable_cache.clear(); signal_engine._bar_score_cache.clear()


@pytest.mark.parametrize("include_ml", [False, True])
def test_y3_backtest_decision_equals_screen_decision_on_every_bar(processed_df, monkeypatch, include_ml):
    _fake_ml(monkeypatch)
    frame = processed_df.iloc[:300]
    decisions = build_v1_decisions(frame, include_ml=include_ml)
    checked = 0
    for index in range(len(frame) - 45, len(frame)):
        screen = generate_stable_signal(frame.iloc[:index + 1], "1d", include_ml=include_ml,
                                        data_is_closed=True, strict_components=True)
        assert decisions[frame.index[index]] == _label(screen.verdict), (index, include_ml)
        checked += 1
    assert checked == 45


def test_y3_decision_does_not_depend_on_history_before_the_stability_window(processed_df):
    """Makine son pencerede sıfırdan kurulur; eski barların ardışık durumu karara sızmaz."""
    frame = processed_df.iloc[:300]
    full = build_v1_decisions(frame, include_ml=False)
    shifted = build_v1_decisions(frame.iloc[40:], include_ml=False)
    common = [d for d in shifted if d in full][-60:]
    assert common and all(full[d] == shifted[d] for d in common)
