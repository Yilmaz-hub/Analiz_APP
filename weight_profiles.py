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
    malformed (caller falls back to DecisionEngineConfig defaults)."""
    asset_class = classify_asset(symbol)
    if asset_class is None:
        return None
    entry = load_profiles().get(asset_class)
    if not entry or "weights" not in entry:
        return None
    weights = entry["weights"]
    if not _valid_weights(weights):
        logger.warning(f"Malformed weight profile for class '{asset_class}', using defaults")
        return None
    return {k: float(weights[k]) for k in WEIGHT_KEYS}
