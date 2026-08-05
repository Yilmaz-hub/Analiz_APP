from signal_engine import _BoundedCache, generate_stable_signal, _bar_score_cache


def test_bounded_cache_evicts_lru_not_everything():
    cache = _BoundedCache(maxsize=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.get("a")  # touch "a" so it's no longer the least-recently-used
    cache.set("d", 4)  # over capacity -> evicts "b" (now the LRU), not everything
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4
    assert cache.get("b") is None


def test_repeated_signal_calls_reuse_bar_score_cache(processed_df):
    """Calling generate_stable_signal on growing slices (simulating
    successive candle closes) should populate the bar-score cache and keep
    reusing entries rather than growing unboundedly per call."""
    _bar_score_cache._data.clear()
    generate_stable_signal(processed_df.iloc[:400], "1d", include_ml=False)
    size_after_first = len(_bar_score_cache._data)
    assert size_after_first > 0

    # One bar further: the replay window mostly overlaps the previous call's,
    # so it should add at most a couple of new entries, not a full new batch.
    generate_stable_signal(processed_df.iloc[:401], "1d", include_ml=False)
    size_after_second = len(_bar_score_cache._data)
    growth = size_after_second - size_after_first
    assert 0 <= growth <= 2, f"expected cache reuse across overlapping windows, got growth={growth}"
