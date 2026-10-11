"""Spec 0023 kanıtı: gerçek app.py'de günlük backtest — BTC (C), SOL (kanal), AAPL (al-tut), BTC "V1 (eski)".

Her varlık geçici, izole bir veri klasörüyle açılır (kullanıcı kayıtlarına dokunmaz); fiyatlar canlı sağlayıcıdan gelir.
Çalıştırma: python docs/evidence/0023/capture.py   (playwright + chromium gerekir)
"""
import os
import subprocess
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CASES = [("btc-kural", "Bitcoin (BTC)", "BTC-USD", "0.00001", "VARLIK", "Binance"),
         ("sol-kanal", "Solana (SOL)", "SOL-USD", "0.001", "VARLIK", "Binance"),
         ("aapl-al-tut", "Apple (AAPL)", "AAPL", "1", "VARLIK", "Yahoo Finance"),
         ("btc-v1-eski", "Bitcoin (BTC)", "BTC-USD", "0.00001", "V1", "Binance")]
DONE = "text=/Toplam Getiri|Al-tut sonucu|Yeterli işlem oluşmadı/"


def _seed(data_dir, name, symbol, mode):
    env = {**os.environ, "ANALIZ_APP_DATA_DIR": data_dir}
    env.pop("ANALIZ_APP_DB_URL", None)
    code = ("import storage, signal_rules as sr; storage.write_doc(storage.ASSETS_KEY, {%r: %r}); "
            "sr.choose_signal_mode(%r)" % (name, symbol, mode))
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)
    return env


def _capture(page, shot, step, source):
    page.goto("http://localhost:8773")
    page.wait_for_selector("text=Periyot:", timeout=180000)
    if source != "Binance":
        page.get_by_text(source, exact=True).click()
        time.sleep(5)
    page.get_by_label("Periyot:").click()
    page.get_by_role("option", name="Günlük").click()
    page.wait_for_selector("text=📊 Backtest: Strateji Performansı", timeout=180000)
    time.sleep(3)
    page.get_by_text("⚙️ İşlem varsayımları").click()
    page.get_by_role("textbox", name="Adet/lot adımı").first.fill(step)
    page.keyboard.press("Enter")
    time.sleep(3)
    page.get_by_text("📊 Backtest: Strateji Performansı").click()
    page.get_by_role("button", name="🚀 Backtest Başlat").click()
    page.wait_for_selector(DONE, timeout=600000)
    time.sleep(4)
    expander = page.locator("[data-testid='stExpander']", has_text="📊 Backtest: Strateji Performansı").first
    expander.screenshot(path=os.path.join(HERE, f"{shot}.png"))


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
        for shot, name, symbol, step, mode, source in CASES:
            env = _seed(tempfile.mkdtemp(prefix="evidence_0023_"), name, symbol, mode)
            srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "app.py"),
                                    "--server.port", "8773", "--server.headless", "true",
                                    "--browser.gatherUsageStats", "false"],
                                   cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                time.sleep(8)
                page = browser.new_page(viewport={"width": 1200, "height": 1600})
                _capture(page, shot, step, source)
                page.close()
                print("ok", shot)
            finally:
                srv.terminate()
                srv.wait()
        browser.close()


if __name__ == "__main__":
    main()
