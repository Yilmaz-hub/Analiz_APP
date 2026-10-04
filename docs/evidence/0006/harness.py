"""Spec 0006 ekran kanıtı: gerçek `app.py`, ağ kaynakları taklit edilmiş, geçici kayıt deposu.

`STATE=partial|prices|asset` ortam değişkeniyle seçilir (bkz. capture.py). Gerçek kayıtlara dokunulmaz.
"""
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
STATE = os.environ.get("STATE", "partial")
os.environ["ANALIZ_APP_DATA_DIR"] = os.path.join(tempfile.gettempdir(), f"analiz_evidence_0006_{STATE}")
os.makedirs(os.environ["ANALIZ_APP_DATA_DIR"], exist_ok=True)

import data_fetchers
import storage
from conftest import make_ohlcv

frame, _ = data_fetchers.process_data(make_ohlcv(), "kanit")
PRICES = {"Chainlink": 18.4, "Hedera": 0, "Bitcoin (BTC)": float(frame["Close"].iloc[-1])}
data_fetchers.get_market_data = lambda *a, **k: (frame, "Binance")
data_fetchers.get_fear_greed_index = lambda: (50, "Neutral")
data_fetchers.get_live_price_for_portfolio = lambda coin, coin_map: PRICES.get(coin, 0)

if not storage.read_doc("seeded"):
    pos = lambda coin, entry, qty: {
        "Coin": coin, "Giriş": entry, "Adet": qty, "Yatırım": entry * qty, "Realized": 0.0,
        "Status": "ACTIVE", "Tarih": "2026-09-01", "Stop": entry * 0.9,
        "Gerçekleşme Zamanı": "2026-09-01T10:00:00+00:00", "V1Verified": True}
    if STATE == "prices":
        storage.write_doc(storage.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD", "Chainlink": "LINKUSD", "Hedera": "HBARUSD"})
        storage.write_doc(storage.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [
            pos("Chainlink", 15.0, 20.0), pos("Hedera", 0.2, 1000.0)]})
    else:
        storage.write_doc(storage.ASSETS_KEY, {"Bitcoin (BTC)": "BTC-USD"})
        storage.write_doc(storage.PORTFOLIO_KEY, {"balance": 1000.0, "positions": [
            pos("Bitcoin (BTC)", 100.0, 10.0)] if STATE == "partial" else []})
    storage.write_doc("seeded", True)

exec(compile(open(os.path.join(ROOT, "app.py"), encoding="utf-8").read(), "app.py", "exec"))
