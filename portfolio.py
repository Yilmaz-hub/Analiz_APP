import json
import os
import time
from config import FileConfig, Constants, RiskConfig
from data_fetchers import get_market_data
from logger import logger
import pandas as pd

def load_portfolio():
    f = FileConfig.PORTFOLIO_FILE
    if os.path.exists(f):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                if 'balance' not in data: data['balance'] = Constants.DEFAULT_PORTFOLIO_BALANCE
                return data
        except Exception as e:
            logger.error(f"Portfolio load error, using defaults: {e}")
            return {"positions": [], "balance": Constants.DEFAULT_PORTFOLIO_BALANCE}
    else:
        return {"positions": [], "history": [], "balance": Constants.DEFAULT_PORTFOLIO_BALANCE}

def save_portfolio(data):
    # Write to a temp file then atomically replace the target, so a crash or
    # a concurrent writer (another tab, the scheduled task) can never leave
    # portfolio.json half-written / corrupted. os.replace() is atomic on
    # both POSIX and Windows.
    target = FileConfig.PORTFOLIO_FILE
    tmp = f"{target}.tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, target)

def validate_portfolio_risk(new_investment, current_balance, open_positions):
    """
    Kelly Criterion ve maksimum pozisyon büyüklüğü kontrolü
    """
    total_equity = current_balance + sum([p.get('Yatırım', 0) for p in open_positions if p.get('Status') == 'ACTIVE'])

    if new_investment > total_equity * RiskConfig.MAX_POSITION_SIZE:
        pct = int(RiskConfig.MAX_POSITION_SIZE * 100)
        return False, f"⚠️ Tek pozisyon toplam varlığın %{pct}'sini aşamaz!"

    total_exposure = sum([p.get('Yatırım', 0) for p in open_positions if p.get('Status') == 'ACTIVE']) + new_investment
    if total_exposure > total_equity * RiskConfig.MAX_TOTAL_EXPOSURE:
        pct = int(RiskConfig.MAX_TOTAL_EXPOSURE * 100)
        return False, f"⚠️ Toplam açık pozisyon %{pct}'yi geçemez!"

    return True, "✅ Risk kabul edilebilir"

def check_active_positions_auto_close(portfolio_data, coin_map):
    """
    Aktif pozisyonları kontrol eder, TP/SL'ye ulaşanları otomatik kapatır
    """
    if "positions" not in portfolio_data: return 0, []
    
    closed_count = 0
    closed_trades = []
    
    from data_fetchers import get_live_price_for_portfolio
    
    for pos in portfolio_data["positions"]:
        if pos.get("Status") == "ACTIVE":
            coin_name = pos.get("Coin")
            tp = pos.get("TP")
            sl = pos.get("SL")
            entry = pos.get("Giris")
            qty = pos.get("Miktar", 0)
            investment = pos.get("Yatırım", 0)
            
            live_price = get_live_price_for_portfolio(coin_name, coin_map)
            
            if live_price > 0:
                if tp and live_price >= tp:
                    profit = (tp - entry) * qty
                    pos['Status'] = 'CLOSED_TP'
                    pos['Exit_Price'] = tp
                    pos['Profit'] = profit
                    pos['Exit_Date'] = time.strftime("%Y-%m-%d %H:%M")
                    portfolio_data['balance'] = portfolio_data.get('balance', 0) + investment + profit
                    closed_count += 1
                    closed_trades.append({
                        'coin': coin_name, 'type': 'TP', 'profit': profit, 'pct': (profit / investment) * 100
                    })
                
                elif sl and live_price <= sl:
                    loss = (sl - entry) * qty
                    pos['Status'] = 'CLOSED_SL'
                    pos['Exit_Price'] = sl
                    pos['Profit'] = loss
                    pos['Exit_Date'] = time.strftime("%Y-%m-%d %H:%M")
                    portfolio_data['balance'] = portfolio_data.get('balance', 0) + investment + loss
                    closed_count += 1
                    closed_trades.append({
                        'coin': coin_name, 'type': 'SL', 'profit': loss, 'pct': (loss / investment) * 100
                    })
    
    if closed_count > 0:
        save_portfolio(portfolio_data)
    
    return closed_count, closed_trades

def multi_timeframe_confirmation(coin_name, symbol, source_pref):
    """
    3 zaman diliminde de aynı yönde sinyal varsa güçlü onay
    """
    signals = {}
    scores = []
    
    from signal_engine import generate_stable_signal  # lazy: avoids import cycle via technical_analysis

    for tf in ["4h", "1d", "1wk"]:
        try:
            df, _ = get_market_data(source_pref, symbol, tf)
            if isinstance(df, pd.DataFrame) and not getattr(df, 'empty', True) and len(df) > 50:  # type: ignore
                status = generate_stable_signal(df, tf).verdict

                if "AL" in status:
                    signals[tf] = "AL"
                    scores.append(1)
                elif "SAT" in status: 
                    signals[tf] = "SAT"
                    scores.append(-1)
                else: 
                    signals[tf] = "NÖTR"
                    scores.append(0)
        except Exception as e:
            logger.debug(f"Multi-TF confirmation failed for {tf}: {e}")
            signals[tf] = "HATA"
            scores.append(0)
    
    if len(scores) == 3:
        if all(s > 0 for s in scores): return "✅ ÜÇ DİLİM AL ONAYI", signals
        elif all(s < 0 for s in scores): return "❌ ÜÇ DİLİM SAT ONAYI", signals
        elif sum(scores) > 0: return "⚠️ KARMA (AL Ağırlıklı)", signals
        elif sum(scores) < 0: return "⚠️ KARMA (SAT Ağırlıklı)", signals
    
    return "📊 Çelişkili Sinyaller", signals
