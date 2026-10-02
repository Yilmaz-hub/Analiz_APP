"""Q6 (spec 0003 AK03 / AC120): hesaplanamayan karar bileşeni nötr sayılmaz."""
import pytest

import signal_engine
from signal_engine import generate_stable_signal, _compute_bar_score
from technical_analysis import build_v1_decisions


@pytest.fixture(autouse=True)
def _fresh_cache():
    signal_engine._stable_cache.clear()
    signal_engine._bar_score_cache.clear()
    yield
    signal_engine._stable_cache.clear()
    signal_engine._bar_score_cache.clear()


def _patch_ml(monkeypatch, result=None, error=None):
    import ml_models

    def fake(_df):
        if error:
            raise error
        return result

    monkeypatch.setattr(ml_models, "calculate_ml_direction_signal", fake)


def test_q6_ml_none_marks_ml_unavailable(processed_df, monkeypatch):
    _patch_ml(monkeypatch, result=None)
    sig = generate_stable_signal(processed_df, "1d", strict_components=True, data_is_closed=True)
    assert sig.unavailable_components == ("ml",)
    assert sig.verdict == "BEKLE"


def test_q6_ml_exception_marks_ml_unavailable(processed_df, monkeypatch):
    _patch_ml(monkeypatch, error=RuntimeError("model yok"))
    sig = generate_stable_signal(processed_df, "1d", strict_components=True, data_is_closed=True)
    assert sig.unavailable_components == ("ml",)


def test_q6_neutral_ml_result_is_not_missing(processed_df, monkeypatch):
    _patch_ml(monkeypatch, result={"direction": "NEUTRAL", "confidence": 50.0, "predicted_change_pct": 0.0})
    sig = generate_stable_signal(processed_df, "1d", strict_components=True, data_is_closed=True)
    assert sig.unavailable_components == ()


def test_q6_pattern_exception_marks_pattern_unavailable(processed_df, monkeypatch):
    _patch_ml(monkeypatch, result={"direction": "NEUTRAL", "confidence": 50.0, "predicted_change_pct": 0.0})

    def boom(_df):
        raise ValueError("formasyon hatasi")

    monkeypatch.setattr(signal_engine, "detect_patterns", boom)
    sig = generate_stable_signal(processed_df, "1d", strict_components=True, data_is_closed=True)
    assert sig.unavailable_components == ("pattern",)


def test_q6_ml_disabled_is_not_counted_as_missing(processed_df):
    sig = generate_stable_signal(processed_df, "1d", include_ml=False,
                                 strict_components=True, data_is_closed=True)
    assert sig.unavailable_components == ()


def test_q6_non_strict_mode_keeps_legacy_behavior(processed_df, monkeypatch):
    _patch_ml(monkeypatch, result=None)
    sig = generate_stable_signal(processed_df, "1d", data_is_closed=True)
    assert sig.unavailable_components == ()


def test_q6_v1_decisions_mark_uncomputable_bar_without_new_action(processed_df, monkeypatch):
    _patch_ml(monkeypatch, result=None)
    decisions = build_v1_decisions(processed_df.iloc[:260], include_ml=True)
    assert decisions and set(decisions.values()) == {"BILESEN_YOK"}


def test_q6_v1_decisions_unchanged_when_components_ready(processed_df):
    decisions = build_v1_decisions(processed_df.iloc[:260], include_ml=False)
    assert "BILESEN_YOK" not in set(decisions.values())
