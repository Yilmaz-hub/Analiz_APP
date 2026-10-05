import os, subprocess, sys, time
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__))
db = "/tmp/evidence_0012.db"
if os.path.exists(db): os.remove(db)
srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", os.path.join(HERE, "harness.py"), "--server.port", "8772",
    "--server.headless", "true", "--browser.gatherUsageStats", "false"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(8)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.environ.get("CHROMIUM", "/opt/pw-browsers/chromium"))
        page = b.new_page(viewport={"width": 1100, "height": 1500})
        page.goto("http://localhost:8772")
        page.wait_for_selector("text=Takip çalışıyor", timeout=90000)
        page.get_by_text("Sanal takip durumunu göster").click()
        page.wait_for_selector("text=Ayrıntı için varlık seçin", timeout=60000)
        time.sleep(2)
        page.screenshot(path=os.path.join(HERE, "sanal-takip.png"), full_page=True)
finally:
    srv.terminate()
