"""Spec 0006 ekran görüntülerini alır (Playwright + önceden kurulu Chromium)."""
import os
import shutil
import subprocess
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMIUM = os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")


def serve(state, port):
    shutil.rmtree(os.path.join(tempfile.gettempdir(), f"analiz_evidence_0006_{state}"), ignore_errors=True)
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "harness.py"), "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        env={**os.environ, "STATE": state}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def settle(page, seconds=4):
    time.sleep(seconds)


def shot(page, name):
    page.screenshot(path=os.path.join(HERE, name), full_page=True)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROMIUM)
    # 1) Kısmi satış: %40 → miktar 4; onay sonrası pozisyon 6 adetle listede kalır (AC17)
    server = serve("partial", 8771)
    try:
        time.sleep(8)
        page = browser.new_page(viewport={"width": 1200, "height": 2600})
        page.goto("http://localhost:8771")
        page.wait_for_selector("text=Kar Al / Satış Yap", timeout=90000)
        page.get_by_text("Kar Al / Satış Yap").click()
        settle(page)
        page.get_by_role("button", name="%25").first.wait_for(timeout=30000)
        page.get_by_role("button", name="%50").first.click()
        settle(page)
        shot(page, "kismi-satis-yuzde.png")
        page.get_by_role("button", name="Satışı Onayla").click()
        settle(page, 6)
        shot(page, "kismi-satis-sonrasi.png")
    finally:
        server.terminate()
    # 2) Fiyat güvenilirliği: LINKUSD canlı fiyatla, HBARUSD fiyat yoksa "fiyat alınamadı" (AC27, AC30)
    server = serve("prices", 8772)
    try:
        time.sleep(8)
        page = browser.new_page(viewport={"width": 1800, "height": 2600})
        page.goto("http://localhost:8772")
        page.wait_for_selector("text=Aktif Pozisyonlar", timeout=90000)
        settle(page, 5)
        heading = page.get_by_text("Aktif Pozisyonlar").first
        heading.scroll_into_view_if_needed()
        time.sleep(1)
        box = heading.bounding_box()
        page.screenshot(path=os.path.join(HERE, "fiyat-guvenilirligi.png"),
                        clip={"x": 300, "y": max(box["y"] - 20, 0), "width": 1400,
                              "height": min(800, 2600 - max(box["y"] - 20, 0))})
    finally:
        server.terminate()
    # 3) Varlık ekleme bilgisi (AC28)
    server = serve("asset", 8773)
    try:
        time.sleep(8)
        page = browser.new_page(viewport={"width": 1200, "height": 2600})
        page.goto("http://localhost:8773")
        page.wait_for_selector("text=Varlık Yönetimi", timeout=90000)
        page.get_by_text("Varlık Yönetimi").click()
        settle(page, 2)
        page.get_by_label("Görünen İsim (Örn: Pound)").fill("Chainlink")
        page.get_by_label("Yahoo Kodu (Örn: GBPUSD=X)").fill("LINKUSD")
        page.get_by_label("Yahoo Kodu (Örn: GBPUSD=X)").press("Tab")
        page.get_by_role("button", name="Listeye Ekle").click()
        settle(page, 6)
        shot(page, "varlik-ekleme.png")
    finally:
        server.terminate()
    browser.close()
