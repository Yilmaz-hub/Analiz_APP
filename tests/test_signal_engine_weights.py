from signal_engine import _compute_bar_score, generate_stable_signal, _weights_fingerprint, _bar_score_cache

CUSTOM_WEIGHTS = {"trend": 0.05, "momentum": 0.05, "volume": 0.60, "pattern": 0.05, "ml": 0.05, "advanced": 0.20}


def test_compute_bar_score_with_custom_weights_differs_from_default(processed_df):
    default_bar = _compute_bar_score(processed_df, "1d", include_ml=False, weights=None)
    custom_bar = _compute_bar_score(processed_df, "1d", include_ml=False, weights=CUSTOM_WEIGHTS)
    assert default_bar is not None and custom_bar is not None
    assert default_bar["score"] != custom_bar["score"]


def test_weights_fingerprint_differs_for_different_weights():
    fp_default = _weights_fingerprint(None)
    fp_a = _weights_fingerprint(CUSTOM_WEIGHTS)
    fp_b = _weights_fingerprint({**CUSTOM_WEIGHTS, "trend": 0.06, "advanced": 0.19})
    assert fp_default is None
    assert fp_a != fp_b


def test_generate_stable_signal_caches_per_weights_not_shared(processed_df):
    """Two calls with different weights on the identical bar must not
    collide in the cache and return each other's signal."""
    _bar_score_cache._data.clear()
    s_default = generate_stable_signal(processed_df, "1d", include_ml=False, weights=None)
    s_custom = generate_stable_signal(processed_df, "1d", include_ml=False, weights=CUSTOM_WEIGHTS)
    assert s_default.final_score != s_custom.final_score
