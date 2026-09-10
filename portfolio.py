import time
from config import Constants, RiskConfig
from data_fetchers import get_market_data
from logger import logger
import pandas as pd

import positions as positions_module
import storage

def empty_portfolio():
    """Hiç kayıt yokken kullanılan başlangıç portföyü (depoya yazılmaz)."""
    return {"positions": [], "history": [], "balance": Constants.DEFAULT_PORTFOLIO_BALANCE}

def load_portfolio():
    """Portföyü depodan okur.

    Erişim hatasında `storage.StorageAccessError` yükselir ve **varsayılan
    portföy döndürülmez** (spec 0004, R8.1): okunamayan portföyü boş ya da
    sıfır bakiyeli göstermek, ardından gelen ilk yazmada gerçek kayıtların
    üzerine yazılmasına yol açıyordu.
    """
    data = storage.read_doc(storage.PORTFOLIO_KEY)
    if data is None:
        return empty_portfolio()
    if not isinstance(data, dict):
        raise storage.StorageAccessError("Portföy kayıtları okunamadı.")
    if 'balance' not in data:
        data['balance'] = Constants.DEFAULT_PORTFOLIO_BALANCE
    if not isinstance(data.get('positions'), list):
        # Pozisyon listesi okunamıyorsa kayıt boşaltılmaz; erişim sorunu olarak
        # bildirilir (R8.3).
        raise storage.StorageAccessError("Portföy kayıtları okunamadı.")
    return data

def save_portfolio(data):
    """Portföyü depoya yazar (tek satırlık atomik upsert)."""
    storage.write_doc(storage.PORTFOLIO_KEY, data)

def reset_portfolio(portfolio_data, confirmed):
    """Portföyü sıfırlar (spec 0004, AC11d). Döner: (başarılı_mı, mesaj, portföy).

    Aktif pozisyon, bekleyen emir veya sorunlu kayıt varsa çalışmaz; her
    durumda kullanıcının ayrıca onayı gerekir.
    """
    current = portfolio_data.get('positions', []) if isinstance(portfolio_data, dict) else []
    reason = positions_module.reset_block_reason(current)
    if reason:
        return False, reason, portfolio_data
    if not confirmed:
        return False, "Sıfırlama için önce onay kutusunu işaretleyin.", portfolio_data
    return True, "Portföy sıfırlandı.", empty_portfolio()

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
    from weight_profiles import get_weights_for_symbol

    for tf in ["4h", "1d", "1wk"]:
        try:
            df, _ = get_market_data(source_pref, symbol, tf)
            if isinstance(df, pd.DataFrame) and not getattr(df, 'empty', True) and len(df) > 50:  # type: ignore
                tf_weights = get_weights_for_symbol(symbol) if tf == "1d" else None
                status = generate_stable_signal(df, tf, weights=tf_weights).verdict

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
