"""Spec 0005 Adım 5 / S3 — aday değerlendirme akışı ve paneli (AC01, AC16, AC17, AC48, AC80, AC85)."""
from datetime import datetime, timezone
from decimal import Decimal

import candidate_ui
import strategy_candidates as sc
from regime_classifier import REGIME_VERSION

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _text(view):
    return " | ".join(view.lines)


def _run(trending_df, decisions=None):
    index = trending_df.index
    v1 = decisions if decisions is not None else {index[260]: "AL", index[280]: "SAT"}
    return candidate_ui.evaluate_candidates(
        "BTC-USD", trending_df, v1, notional=Decimal("1000"), capital=Decimal("10000"),
        quantity_step=Decimal("0.001"), costs=None, now=NOW)


def test_ac01_panel_shows_v1_as_active_and_no_candidate_selected():
    """AC01 — Panel ilk açılışta aktif stratejiyi mevcut V1 olarak gösterir; aday seçili değildir."""
    view = candidate_ui.build_candidate_view(None)
    assert "Aktif strateji: V1 (mevcut strateji)" in _text(view)
    assert "Henüz değerlendirilmiş aday yok." in _text(view)


def test_ac80_breakout_and_pullback_get_separate_records_and_results(trending_df):
    """AC80 — Değerlendirme sonrası kırılım ve geri çekilme ayrı kayıt ve ayrı sonuçla listelenir."""
    outcome = _run(trending_df)
    assert outcome.ok, outcome.reason
    names = [entry["name"] for entry in sc.history()]
    assert sorted(names) == ["Geri çekilme (EMA20, %1)", "Kırılım (20 gün)"]
    text = _text(candidate_ui.build_candidate_view(None))
    assert "Kırılım (20 gün)" in text and "Geri çekilme (EMA20, %1)" in text


def test_ac17_evaluation_never_changes_active_strategy(trending_df):
    """AC17 — Değerlendirme hangi sonucu verirse versin aktif strateji kullanıcı seçimi olmadan değişmez."""
    _run(trending_df)
    assert sc.active_strategy() == "V1"
    assert sc.history()                                     # değerlendirme gerçekten çalıştı ve kayıt bıraktı
    assert any(entry["status"] in (sc.OLCUTU_KARSILADI, sc.OLCUTU_KARSILAMADI, sc.YETERSIZ_VERI)
               for entry in sc.history())


def test_ac16_failed_candidate_is_visible_in_history_view():
    """AC16 — Ölçütü karşılamayan aday panelde başarısız sonucu ile görünür."""
    candidate = sc.default_candidates("KRIPTO", None, None)[0]
    sc.preregister(candidate, NOW)
    sc.record_result(candidate, "KRIPTO", sc.Verdict(sc.OLCUTU_KARSILAMADI, "Getiri düşük."), NOW)
    text = _text(candidate_ui.build_candidate_view(None))
    assert "Kırılım (20 gün) — KRIPTO: Ölçütü karşılamadı" in text


def test_ac48_trial_result_is_labelled_not_approved():
    """AC48 — Onaysız kurallı adayın sonucu "deneme" olarak, onaylı değerlendirme gibi gösterilmez."""
    candidate = sc.default_candidates("KRIPTO", None, None)[0]
    sc.record_result(candidate, "KRIPTO", sc.Verdict(sc.DENEME, ""), NOW)
    text = _text(candidate_ui.build_candidate_view(None))
    assert "Deneme — onaylı strateji değerlendirmesi değildir" in text


def test_ac85_panel_shows_market_condition_with_classifier_version(trending_df):
    """AC85 — Panelde gösterilen piyasa koşulu, sınıflandırıcı sürümüyle birlikte görünür."""
    view = candidate_ui.build_candidate_view(trending_df)
    assert f"({REGIME_VERSION})" in _text(view)
    assert "Son günün piyasa koşulu:" in _text(view)


def test_short_history_refuses_without_recording(trending_df):
    """Değerlendirme dilimi kurulamayacak kadar kısa veri: koşum yapılmaz, geçmişe yazılmaz."""
    outcome = candidate_ui.evaluate_candidates(
        "BTC-USD", trending_df.iloc[:2], {}, notional=Decimal("1000"), capital=Decimal("10000"),
        quantity_step=Decimal("0.001"), costs=None, now=NOW)
    assert not outcome.ok and sc.history() == []
