"""Spec 0007 AC10 — gerçek tarayıcıda satış paneli yerel çalışır (Playwright + Chromium gerekir).

`python -m pytest tests/ -m browser -v` ile koşulur; Playwright kurulu değilse atlanır.
Harness: `docs/evidence/0006/harness.py` (gerçek `app.py`, taklit ağ, geçici kayıt deposu).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

pytestmark = pytest.mark.browser
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMIUM = os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")


def test_ac10_typing_in_the_sell_panel_does_not_recompute_the_page(tmp_path):
    """AC10 — Satış fiyatı ve yüzde yazılınca grafik verisi yeniden çağrılmaz; satış onayında sayfa bir kez tümüyle yenilenir."""
    counter = tmp_path / "calls.txt"
    counter.write_text("")
    shutil.rmtree(os.path.join(tempfile.gettempdir(), "analiz_evidence_0006_partial"), ignore_errors=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "docs", "evidence", "0006", "harness.py"),
         "--server.port", "8791", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        env={**os.environ, "STATE": "partial", "COUNT_FILE": str(counter)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    calls = lambda: len(counter.read_text().split())
    try:
        time.sleep(8)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=CHROMIUM if os.path.exists(CHROMIUM) else None)
            page = browser.new_page(viewport={"width": 1200, "height": 2600})
            page.goto("http://localhost:8791")
            page.wait_for_selector("text=Kar Al / Satış Yap", timeout=90000)
            page.get_by_text("Kar Al / Satış Yap").click()
            time.sleep(3)
            baseline = calls()
            assert baseline > 0
            price = page.get_by_label("Satış Fiyatı")
            price.click(); price.fill("15"); price.press("Enter")
            time.sleep(3)
            percent = page.get_by_label("Yüzde (%)")
            percent.click(); percent.fill("40"); percent.press("Enter")
            time.sleep(3)
            assert calls() == baseline                                  # sayfa yeniden hesaplanmadı
            assert page.get_by_label("Satılan miktar (adet)").input_value() == "4.00000000"
            assert page.locator('[data-testid="stException"]').count() == 0
            page.get_by_role("button", name="Satışı Onayla").click()
            time.sleep(6)
            assert calls() > baseline                                   # onaydan sonra sayfa yenilendi
            browser.close()
    finally:
        server.terminate()
