"""Spec 0005 Adım 7 / S6 — ekrandan bağımsız koşucu (AC35, AC86)."""
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import forward_tracker as ft
from conftest import make_ohlcv

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path):
    frame = make_ohlcv()
    directory = tmp_path / "fixture"
    directory.mkdir()
    frame.to_csv(directory / "BTC-USD.csv")
    return directory, [ts.date() for ts in frame.index]


def _run(directory, now):
    return subprocess.run(
        [sys.executable, "-m", "forward_runner", "--now", now.isoformat(), "--fixture",
         str(directory), "--no-ml", "--symbols", "BTC-USD"],
        cwd=ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=300)


def test_ac35_headless_runner_records_within_two_hours_without_ui(tmp_path):
    """AC35 — Arayüz kapalıyken koşucu süreci yeni günün kararını 2 saat içinde zamanında kaydeder."""
    directory, days = _fixture(tmp_path)
    last = days[-1]
    now = ft.available_at("KRIPTO", last) + timedelta(hours=1, minutes=59)
    done = _run(directory, now)
    assert done.returncode == 0, done.stderr
    # Çalışan bir Streamlit oturumu yoktur; kütüphanenin önbellek uyarısı bunu doğrular.
    assert "Traceback" not in done.stderr
    (version,) = ft.versions("BTC-USD")
    assert version.endswith("/ML-YOK/TATBIKAT")      # --now: hızlandırılmış, gerçek kayıttan ayrı sürüm
    stored = ft.get_decision("BTC-USD", version, last)
    assert stored is not None and stored.on_time is True
    assert stored.evaluated_at == now
    assert stored.real_clock is False  # sabitlenmiş saat: yeterlilik sayacına girmez (AC87)


def test_ac35_missed_days_are_backfilled_and_marked_late(tmp_path):
    """AC35 — Çalışılmayan günler sonraki çalışmada tamamlanır ve "sonradan oluşturuldu" işaretlenir."""
    directory, days = _fixture(tmp_path)
    first_now = ft.available_at("KRIPTO", days[-4]) + timedelta(minutes=30)
    assert _run(directory, first_now).returncode == 0
    later = ft.available_at("KRIPTO", days[-1]) + timedelta(minutes=30)
    done = _run(directory, later)
    assert done.returncode == 0, done.stderr
    (version,) = ft.versions("BTC-USD")
    statuses = {d.candle_day: ft.status_text(d) for d in ft.decisions("BTC-USD", version)}
    assert ft.get_decision("BTC-USD", version, days[-4]).on_time is True
    assert [ft.get_decision("BTC-USD", version, d).on_time for d in days[-3:-1]] == [False, False]
    assert ft.get_decision("BTC-USD", version, days[-1]).on_time is True
    assert len(statuses) == 4


def test_ac86_runner_reports_provider_wait_separately(tmp_path):
    """AC86 — Koşucu raporunda sağlayıcı bekleme süresi hesaplama süresinden ayrı gösterilir."""
    directory, days = _fixture(tmp_path)
    done = _run(directory, ft.available_at("KRIPTO", days[-1]) + timedelta(minutes=5))
    assert "sağlayıcı bekleme:" in done.stdout and "hesaplama:" in done.stdout
