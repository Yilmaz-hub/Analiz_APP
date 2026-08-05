import weight_profiles as wp


def test_classify_asset_crypto():
    assert wp.classify_asset("BTC-USD") == "crypto"
    assert wp.classify_asset("DOGE-USDT") == "crypto"


def test_classify_asset_bist():
    assert wp.classify_asset("THYAO.IS") == "bist"


def test_classify_asset_forex():
    assert wp.classify_asset("EURUSD=X") == "forex"


def test_classify_asset_commodity():
    assert wp.classify_asset("XAU_GOLD") == "commodity"
    assert wp.classify_asset("GRAM_TRY") == "commodity"


def test_classify_asset_unmatched_returns_none():
    assert wp.classify_asset("AAPL") is None
    assert wp.classify_asset("") is None
    assert wp.classify_asset(None) is None


def test_load_profiles_missing_file_returns_empty_dict(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(missing))
    assert wp.load_profiles() == {}


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    data = {"crypto": {"weights": {k: 1 / 6 for k in wp.WEIGHT_KEYS}}}
    wp.save_profiles(data)
    assert target.exists()
    assert wp.load_profiles() == data


def test_get_weights_for_symbol_returns_valid_profile(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    weights = {"trend": 0.3, "momentum": 0.2, "volume": 0.15, "pattern": 0.1, "ml": 0.1, "advanced": 0.15}
    wp.save_profiles({"crypto": {"weights": weights, "test_score": 8.8}})
    assert wp.get_weights_for_symbol("BTC-USD") == weights


def test_get_weights_for_symbol_unmatched_class_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    wp.save_profiles({})
    assert wp.get_weights_for_symbol("AAPL") is None


def test_get_weights_for_symbol_no_profile_for_class_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    wp.save_profiles({})
    assert wp.get_weights_for_symbol("BTC-USD") is None


def test_get_weights_for_symbol_malformed_profile_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    bad_weights = {"trend": 0.5, "momentum": 0.5, "volume": 0.5, "pattern": 0.5, "ml": 0.5, "advanced": 0.5}
    wp.save_profiles({"crypto": {"weights": bad_weights}})  # sums to 3.0, not 1.0
    assert wp.get_weights_for_symbol("BTC-USD") is None


def test_get_weights_for_symbol_non_dict_entry_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    wp.save_profiles({"crypto": 5})  # malformed: entry isn't even a dict
    assert wp.get_weights_for_symbol("BTC-USD") is None


def test_get_weights_for_symbol_failed_validation_returns_none(tmp_path, monkeypatch):
    # Mirrors the real committed profile that triggered this fix: a tiny
    # sanity-check tuning run whose held-out test_score was REJECT_SCORE.
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    weights = {"trend": 0.3, "momentum": 0.2, "volume": 0.15, "pattern": 0.1, "ml": 0.1, "advanced": 0.15}
    wp.save_profiles({"crypto": {"weights": weights, "train_score": 8.8, "test_score": -1000000.0}})
    assert wp.get_weights_for_symbol("BTC-USD") is None


def test_get_weights_for_symbol_zero_test_score_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    weights = {"trend": 0.3, "momentum": 0.2, "volume": 0.15, "pattern": 0.1, "ml": 0.1, "advanced": 0.15}
    wp.save_profiles({"crypto": {"weights": weights, "test_score": 0}})
    assert wp.get_weights_for_symbol("BTC-USD") is None


def test_get_weights_for_symbol_missing_test_score_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    weights = {"trend": 0.3, "momentum": 0.2, "volume": 0.15, "pattern": 0.1, "ml": 0.1, "advanced": 0.15}
    wp.save_profiles({"crypto": {"weights": weights}})  # no test_score key at all
    assert wp.get_weights_for_symbol("BTC-USD") is None


def test_get_weights_for_symbol_non_numeric_test_score_returns_none(tmp_path, monkeypatch):
    target = tmp_path / "weight_profiles.json"
    monkeypatch.setattr(wp.FileConfig, "WEIGHT_PROFILES_FILE", str(target))
    weights = {"trend": 0.3, "momentum": 0.2, "volume": 0.15, "pattern": 0.1, "ml": 0.1, "advanced": 0.15}
    wp.save_profiles({"crypto": {"weights": weights, "test_score": "n/a"}})
    assert wp.get_weights_for_symbol("BTC-USD") is None
