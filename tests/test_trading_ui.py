from datetime import datetime, timezone
from decimal import Decimal

from trade_decisions import Position, PositionState, Signal
from trading_ui import PanelInput, aggregate_totals, build_decision_panel, paper_defaults, validate_delay, validate_fee


UTC = timezone.utc
NOW = datetime(2026, 9, 10, tzinfo=UTC)


def _input(**changes):
    values = dict(signal=Signal.WAIT, position=Position.flat(), data_status="GECERLI",
                  fee=Decimal("0"), spread_bps=Decimal("0"), slippage_bps=Decimal("0"),
                  confidence=Decimal("60"), decision_at=NOW)
    values.update(changes)
    return PanelInput(**values)


def _open():
    return Position(PositionState.OPEN, Decimal("2"), Decimal("100"), NOW, Decimal("90"))


def test_requirement_unknown_position_default():
    assert build_decision_panel(_input(position=Position(PositionState.UNKNOWN))).action is None


def test_requirement_sell_full_exit():
    assert build_decision_panel(_input(signal=Signal.SELL, position=_open())).action == "TAMAMINI SAT"


def test_ac15_missing_component_warning():
    panel = build_decision_panel(_input(missing_component="ML"))
    assert "ML" in panel.messages and not panel.comparable


def test_ac19_empty_data_state():
    panel = build_decision_panel(_input(data_status="VERI_YOK"))
    assert panel.action is None and "VERI_YOK" in panel.messages


def test_ac26_negative_fee_message():
    assert validate_fee(Decimal("-1")) == "GECERSIZ_KOMISYON"


def test_ac27_unknown_fee_no_verified_net():
    assert not build_decision_panel(_input(fee=None)).net_verified


def test_ac33_exit_reason_visible():
    assert build_decision_panel(_input(position=_open(), exit_reason="STOP")).exit_reason == "STOP"


def test_ac34_confidence_not_probability():
    assert "kazanma olasılığı değildir" in build_decision_panel(_input()).confidence_note


def test_ac35_assumptions_visible():
    panel = build_decision_panel(_input(spread_bps=Decimal("20"), slippage_bps=Decimal("5")))
    assert panel.assumptions == ("Makas: 20 bp", "Kayma: 5 bp")


def test_ac37_unknown_no_unconditional_buy():
    panel = build_decision_panel(_input(signal=Signal.BUY, position=Position(PositionState.UNKNOWN)))
    assert panel.action is None and "POZISYON_BILINMIYOR" in panel.messages


def test_ac38_zero_open_quantity_invalid():
    invalid = Position(PositionState.OPEN, Decimal("0"), Decimal("100"), NOW, Decimal("90"))
    assert "GECERSIZ_POZISYON" in build_decision_panel(_input(position=invalid)).messages


def test_ac54_ambiguous_sequence_label():
    assert "SIRA_BILINMIYOR" in build_decision_panel(_input(ambiguous_sequence=True)).messages


def test_ac58_negative_delay_rejected():
    assert validate_delay(-1) == "GECERSIZ_TEPKI_SURESI"


def test_ac64_open_position_data_risk():
    panel = build_decision_panel(_input(position=_open(), data_status="ESKI_VERI"))
    assert "GUNCEL_RISK_DEGERLENDIRILEMIYOR" in panel.messages


def test_ac65_old_decision_timestamp():
    panel = build_decision_panel(_input(data_status="ESKI_VERI", decision_at=NOW))
    assert panel.old_decision_at == NOW and "ESKI_KARAR" in panel.messages


def test_ac71_out_of_scope_label():
    assert "V1_DOGRULANMADI" in build_decision_panel(_input(in_scope=False)).messages


def test_ac72_unknown_without_interaction():
    assert build_decision_panel(_input(position=Position(PositionState.UNKNOWN))).messages[0] == "POZISYON_BILINMIYOR"


def test_ac76_no_invented_stop_raise():
    panel = build_decision_panel(_input(position=_open(), suggested_stop=Decimal("98")))
    assert panel.stop == Decimal("90") and "SABIT_STOP_SAT" in panel.messages


def test_ac79_real_and_simulated_labels():
    panel = build_decision_panel(_input())
    assert (panel.real_label, panel.paper_label) == ("TEYITLI GERCEK", "SANAL STRATEJI")


def test_ac80_record_trade_unknown_fee():
    panel = build_decision_panel(_input(fee=None, real_trade_confirmed=True))
    assert panel.trade_accepted and not panel.net_verified


def test_ac82_no_next_bar_pending():
    panel = build_decision_panel(_input(next_bar_available=False, signal=Signal.BUY))
    assert panel.pending and "SONRAKI_MUM_BEKLENIYOR" in panel.messages


def test_ac86_provisional_loss_classification():
    panel = build_decision_panel(_input(fee=None, gross_result=Decimal("-1")))
    assert panel.provisional_result == "GECICI_ZARAR" and not panel.net_verified


def test_ac90_paper_defaults():
    assert paper_defaults("USD") == (Decimal("10000"), Decimal("1000"), "USD")


def test_ac93_no_mixed_currency_total():
    result = aggregate_totals({"USD": Decimal("2"), "USDT": Decimal("3"), "TRY": Decimal("4")})
    assert result == {"USD/USDT": Decimal("5"), "TRY": Decimal("4")}


def test_ac96_unknown_fee_cash_unverified():
    assert "NAKIT_YETERLILIGI_DOGRULANMADI" in build_decision_panel(_input(fee=None)).messages


def test_ac97_unknown_spread_slippage_labels():
    panel = build_decision_panel(_input(spread_bps=None, slippage_bps=None))
    assert "MAKAS_BILINMIYOR" in panel.messages and "KAYMA_BILINMIYOR" in panel.messages


def test_ac101_below_minimum_quantity():
    panel = build_decision_panel(_input(quantity_error="ASGARI_MIKTAR"))
    assert panel.action is None and "ASGARI_MIKTAR" in panel.messages


def test_ac102_unknown_quantity_step():
    panel = build_decision_panel(_input(quantity_error="MIKTAR_ADIMI_BILINMIYOR"))
    assert panel.action is None and "MIKTAR_ADIMI_BILINMIYOR" in panel.messages


def test_ac105_gold_reference():
    panel = build_decision_panel(_input(asset_kind="XAU"))
    assert "GC=F VADELI ALTIN REFERANSI" in panel.messages


def test_ac106_derived_gold():
    panel = build_decision_panel(_input(asset_kind="GRAM_TRY"))
    assert not panel.stop_simulation_verified and "TURETILMIS_OHLC" in panel.messages


def test_ac113_fee_boundary():
    assert validate_fee(Decimal("0")) is None
    assert validate_fee(Decimal("100")) == "GECERSIZ_KOMISYON"
    assert validate_fee(Decimal("NaN")) == "GECERSIZ_KOMISYON"


def test_ac120_component_failure():
    failed = build_decision_panel(_input(signal=Signal.BUY, missing_component="volume"))
    neutral = build_decision_panel(_input(signal=Signal.BUY, neutral_components=("volume",)))
    assert failed.action is None and "volume" in failed.messages and neutral.action == "SATIN AL"
