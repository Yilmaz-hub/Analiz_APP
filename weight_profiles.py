"""
Per-asset-class signal weight profiles: classification, storage, and
lookup. Written/updated offline by optimize_weights.py; read live by
app.py, scanner.py, paper_trading.py, and portfolio.py.
"""
import json
import os

from config import FileConfig
from logger import logger

WEIGHT_KEYS = ("trend", "momentum", "volume", "pattern", "ml", "advanced")
ASSET_CLASSES = ("crypto", "bist", "commodity", "forex")


def classify_asset(symbol):
    """Map a ticker symbol to one of ASSET_CLASSES, or None if it doesn't
    match any known pattern (caller falls back to shipped defaults)."""
    if not symbol:
        return None
    if symbol.endswith(".IS"):
        return "bist"
    if symbol.endswith("=X"):
        return "forex"
    if symbol in ("XAU_GOLD", "GRAM_TRY"):
        return "commodity"
    if "-USD" in symbol or "-USDT" in symbol:
        return "crypto"
    return None


def load_profiles():
    f = FileConfig.WEIGHT_PROFILES_FILE
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception as e:
            logger.error(f"Weight profiles load error: {e}")
    return {}


def save_profiles(data):
    target = FileConfig.WEIGHT_PROFILES_FILE
    tmp = f"{target}.tmp"
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def _valid_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_KEYS):
        return False
    try:
        total = sum(float(weights[k]) for k in WEIGHT_KEYS)
    except (TypeError, ValueError):
        return False
    return abs(total - 1.0) < 1e-6


def get_weights_for_symbol(symbol):
    """Return the tuned weight dict for symbol's asset class, or None if
    no class matches / no profile exists / the stored profile is
    malformed / the profile's validation score doesn't clear the bar
    (caller falls back to DecisionEngineConfig defaults in every case)."""
    asset_class = classify_asset(symbol)
    if asset_class is None:
        return None
    entry = load_profiles().get(asset_class)
    if not isinstance(entry, dict) or "weights" not in entry:
        return None
    weights = entry["weights"]
    if not _valid_weights(weights):
        logger.warning(f"Malformed weight profile for class '{asset_class}', using defaults")
        return None
    test_score = entry.get("test_score")
    if not isinstance(test_score, (int, float)) or isinstance(test_score, bool) or test_score <= 0:
        logger.warning(f"Weight profile for class '{asset_class}' failed held-out validation "
                       f"(test_score={test_score}), using defaults")
        return None
    walk_forward_scores = entry.get("walk_forward_scores")
    if walk_forward_scores is not None:
        valid_scores = (
            isinstance(walk_forward_scores, list)
            and len(walk_forward_scores) >= 2
            and all(isinstance(score, (int, float)) and not isinstance(score, bool)
                    and score > 0 for score in walk_forward_scores)
        )
        if not valid_scores:
            logger.warning(f"Weight profile for class '{asset_class}' failed walk-forward validation, using defaults")
            return None
    return {k: float(weights[k]) for k in WEIGHT_KEYS}
