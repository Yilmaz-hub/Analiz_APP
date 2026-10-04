"""Spec 0005 Adım 5 / S3 — aday kaydı, parmak izi, Q03 ölçütleri, geçmiş ve aktif strateji."""
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

import strategy_candidates as sc
from regime_classifier import YATAY, YUKSELEN

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
BREAKOUT = sc.Candidate("Kırılım 20", "KIRILIM", "V1", "KRIPTO", {"lookback": "20"},
                        date(2025, 1, 1), date(2026, 1, 1))
PULLBACK = sc.Candidate("Geri çekilme EMA20", "GERI_CEKILME", "V1", "KRIPTO",
                        {"ema": "20", "tolerance_pct": "1"}, date(2025, 1, 1), date(2026, 1, 1))
REF = sc.Metrics(net_return_pct=Decimal("10"), max_drawdown_pct=Decimal("20"),
                 expectancy=Decimal("5"), closed_count=40)


def _metrics(**changes):
    return replace(REF, **changes)


def test_ac01_default_active_strategy_is_v1():
    """AC01 — İlk açılışta aktif strateji mevcut V1'dir; hiçbir aday seçili değildir."""
    assert sc.active_strategy() == sc.REFERENCE_STRATEGY == "V1"


def test_ac17_passing_candidate_does_not_switch_active_strategy():
    """AC17 — Aday tüm ölçütleri sağlasa da kullanıcı seçimi olmadan aktif strateji değişmez."""
    sc.preregister(BREAKOUT, NOW)
    verdict = sc.judge(_metrics(net_return_pct=Decimal("15")), REF)
    assert verdict.status == sc.OLCUTU_KARSILADI
    sc.record_result(BREAKOUT, "KRIPTO", verdict, NOW)
    assert sc.active_strategy() == "V1"
    sc.choose_active_strategy(BREAKOUT)  # yalnız açık kullanıcı eylemi değiştirir
    assert sc.active_strategy() == sc.strategy_fingerprint(BREAKOUT)


def test_ac18_29_closed_trades_is_insufficient_even_if_return_is_higher():
    """AC18 — 29 kapanmış işlemli aday getirisi üstün olsa bile yeterli değerlendirme sayılmaz."""
    verdict = sc.judge(_metrics(net_return_pct=Decimal("50"), closed_count=29), REF)
    assert verdict.status == sc.YETERSIZ_VERI


def test_ac19_30_closed_trades_is_not_blocked_by_count():
    """AC19 — 30 kapanmış işlemli aday yalnız işlem sayısı nedeniyle engellenmez."""
    verdict = sc.judge(_metrics(net_return_pct=Decimal("11"), closed_count=30), REF)
    assert verdict.status == sc.OLCUTU_KARSILADI


def test_ac20_equal_return_is_not_higher():
    """AC20 — Aday getirisi referansa eşitse “daha yüksek getiri” koşulu sağlanmaz."""
    verdict = sc.judge(_metrics(), REF)
    assert verdict.status == sc.OLCUTU_KARSILAMADI
    assert "getiri" in verdict.reason.lower()


def test_ac21_higher_return_with_bigger_drawdown_fails():
    """AC21 — Getirisi yüksek ama maksimum düşüşü referanstan büyük aday ölçütü karşılamaz."""
    verdict = sc.judge(_metrics(net_return_pct=Decimal("30"), max_drawdown_pct=Decimal("20.01")), REF)
    assert verdict.status == sc.OLCUTU_KARSILAMADI


def test_ac51_equal_drawdown_does_not_block():
    """AC51 — Adayın maksimum düşüşü referansa tam eşitse bu ölçüt nedeniyle engellenmez."""
    verdict = sc.judge(_metrics(net_return_pct=Decimal("11"), max_drawdown_pct=Decimal("20")), REF)
    assert verdict.status == sc.OLCUTU_KARSILADI


def test_ac52_zero_expectancy_is_not_positive():
    """AC52 — İşlem beklentisi tam 0 olan aday pozitif beklenti koşulunu sağlamaz."""
    verdict = sc.judge(_metrics(net_return_pct=Decimal("11"), expectancy=Decimal("0")), REF)
    assert verdict.status == sc.OLCUTU_KARSILAMADI


def test_ac22_each_market_is_judged_separately():
    """AC22 — Kriptoda yeterli, BIST'te yetersiz aday için BIST sonucu yeterli gösterilmez."""
    verdicts = sc.judge_by_market({
        "KRIPTO": (_metrics(net_return_pct=Decimal("11")), REF),
        "BIST": (_metrics(net_return_pct=Decimal("11"), closed_count=12), REF),
    })
    assert verdicts["KRIPTO"].status == sc.OLCUTU_KARSILADI
    assert verdicts["BIST"].status == sc.YETERSIZ_VERI


def test_ac48_unapproved_rule_is_only_a_trial():
    """AC48 — Kuralı onaylanmamış adayın sonucu onaylı strateji değerlendirmesi olarak yayımlanmaz."""
    trial = replace(BREAKOUT, exit_rule="ONAYSIZ_CIKIS")
    assert not sc.is_approved(trial)
    verdict = sc.judge(_metrics(net_return_pct=Decimal("50")), REF, approved=sc.is_approved(trial))
    assert verdict.status == sc.DENEME
    assert sc.is_approved(BREAKOUT) and sc.is_approved(PULLBACK)


def test_ac69_run_requires_preregistration():
    """AC69 — Ölçütleri ve dönemi önceden kaydedilmemiş aday için koşum başlatılmaz."""
    permit = sc.start_run(BREAKOUT)
    assert not permit.ok and "önceden kaydedilmemiş" in permit.reason
    sc.preregister(BREAKOUT, NOW)
    assert sc.start_run(BREAKOUT).ok
    changed = replace(BREAKOUT, period_end=date(2026, 6, 1))
    assert not sc.start_run(changed).ok  # dönem değişince ön kayıt geçmez


def test_ac79_setting_change_is_new_version_without_inherited_count():
    """AC79 — Bir ayar değişince sürüm değişir; önceki sürümün gözlem sayacı devralınmaz."""
    sc.preregister(BREAKOUT, NOW)
    sc.record_result(BREAKOUT, "KRIPTO", sc.judge(_metrics(), REF), NOW)
    tuned = replace(BREAKOUT, settings={"lookback": "30"})
    assert sc.fingerprint(tuned) != sc.fingerprint(BREAKOUT)
    assert sc.fingerprint(replace(BREAKOUT, name="Başka ad")) == sc.fingerprint(BREAKOUT)
    assert sc.observation_count(BREAKOUT) == 1
    assert sc.observation_count(tuned) == 0


def test_ac16_failed_candidate_stays_in_history():
    """AC16 — Ölçütleri karşılamayan kaydedilmiş aday değerlendirme geçmişinde başarısız sonucuyla görünür."""
    sc.preregister(BREAKOUT, NOW)
    sc.record_result(BREAKOUT, "KRIPTO", sc.judge(_metrics(), REF), NOW)
    entries = sc.history()
    assert [(e["name"], e["status"]) for e in entries] == [("Kırılım 20", sc.OLCUTU_KARSILAMADI)]


def test_ac15_single_filter_label_only_when_one_setting_differs():
    """AC15 — Diğer ayarları da farklı adaylar tek filtre etkisi sonucu olarak sunulmaz."""
    one = replace(PULLBACK, settings={"ema": "20", "tolerance_pct": "2"})
    two = replace(PULLBACK, settings={"ema": "50", "tolerance_pct": "2"})
    assert sc.single_filter_effect(PULLBACK, one) == "tolerance_pct"
    assert sc.single_filter_effect(PULLBACK, two) is None
    assert sc.single_filter_effect(PULLBACK, replace(one, market="BIST")) is None


def _entry_frame():
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    # gün 3: kapanış önceki en yüksek kapanışı aşar (kırılım);
    # gün 5: düşük EMA20'ye yaklaşır, kapanış üstte (geri çekilme)
    close = np.array([100, 101, 102, 110, 108, 109], dtype=float)
    low = np.array([99, 100, 101, 109, 104, 100], dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": low, "Close": close}, index=index)


def test_ac80_breakout_and_pullback_are_separate_candidates_with_separate_results():
    """AC80 — Kırılım ve geri çekilme adayları ayrı kayıt ve ayrı sonuçla listelenir."""
    frame = _entry_frame()
    regimes = pd.Series([YUKSELEN] * 6, index=frame.index, dtype=object)
    v1 = {frame.index[4]: "SAT"}
    # B12: ayarlar adayın kendisinde; kısa veri için aday ayarları küçültülür.
    breakout = sc.candidate_decisions(frame, v1, replace(BREAKOUT, settings={"lookback": "3"}), regimes)
    pullback = sc.candidate_decisions(
        frame, v1, replace(PULLBACK, settings={"ema": "3", "tolerance_pct": "1"}), regimes)
    assert breakout[frame.index[3]] == "AL" and pullback[frame.index[3]] != "AL"
    assert pullback[frame.index[5]] == "AL" and breakout[frame.index[5]] != "AL"
    assert breakout[frame.index[4]] == pullback[frame.index[4]] == "SAT"  # çıkış V1 ile aynı

    for candidate in (BREAKOUT, PULLBACK):
        sc.preregister(candidate, NOW)
        sc.record_result(candidate, "KRIPTO", sc.judge(_metrics(), REF), NOW)
    names = {e["fingerprint"]: e["name"] for e in sc.history()}
    assert names == {sc.fingerprint(BREAKOUT): "Kırılım 20", sc.fingerprint(PULLBACK): "Geri çekilme EMA20"}


def test_ac80_candidate_entries_only_in_rising_market():
    """AC80 — Aday girişleri yalnız yükselen piyasa gününde üretilir (Q02)."""
    frame = _entry_frame()
    flat = pd.Series([YATAY] * 6, index=frame.index, dtype=object)
    decisions = sc.candidate_decisions(frame, {}, replace(BREAKOUT, settings={"lookback": "3"}), flat)
    assert "AL" not in decisions.values()
