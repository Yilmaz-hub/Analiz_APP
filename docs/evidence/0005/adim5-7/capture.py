"""Kanıt sayfasının her durumunun ekran görüntüsünü alır (Playwright + önceden kurulu Chromium)."""
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = ["regime", "candidates", "risk", "protection", "forward"]
PORT = "8765"
server = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "harness.py"), "--server.port", PORT,
     "--server.headless", "true", "--browser.gatherUsageStats", "false"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(8)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"))
        page = browser.new_page(viewport={"width": 900, "height": 700})
        for state in STATES:
            page.goto(f"http://localhost:{PORT}/?state={state}")
            page.wait_for_selector("h3", timeout=60000)
            time.sleep(2)
            page.screenshot(path=os.path.join(HERE, f"{state}.png"), full_page=True)
        browser.close()
finally:
    server.terminate()
