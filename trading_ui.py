from dataclasses import dataclass
from decimal import Decimal

from trade_decisions import PositionState, Signal, validate_position


@dataclass(frozen=True)
class PanelInput:
    signal: Signal
    position: object
    data_status: str
    fee: Decimal | None
    spread_bps: Decimal | None
    slippage_bps: Decimal | None
    confidence: Decimal
    decision_at: object
    missing_component: str | None = None
    exit_reason: str | None = None
    ambiguous_sequence: bool = False
    in_scope: bool = True
    suggested_stop: Decimal | None = None
    real_trade_confirmed: bool = False
    next_bar_available: bool = True
    gross_result: Decimal | None = None
    quantity_error: str | None = None
    asset_kind: str | None = None
    neutral_components: tuple = ()


@dataclass(frozen=True)
class DecisionPanel:
    action: str | None
    messages: tuple
    comparable: bool
    net_verified: bool
    exit_reason: str | None
    confidence_note: str
    assumptions: tuple
    old_decision_at: object | None
    stop: Decimal | None
    real_label: str
    paper_label: str
    trade_accepted: bool
    pending: bool
    provisional_result: str | None
    stop_simulation_verified: bool


def _display_decimal(value):
    return format(Decimal(value), "f")


def build_decision_panel(value):
    messages = []
    action = None
    validation = validate_position(value.position)
    valid_context = True

    if value.position.state is PositionState.UNKNOWN:
        messages.append("POZISYON_BILINMIYOR")
        valid_context = False
    elif value.position.state is PositionState.INVALID or not validation.is_valid:
        messages.append("GECERSIZ_POZISYON")
        valid_context = False
    if value.data_status != "GECERLI":
        messages.append(value.data_status)
        messages.append("ESKI_KARAR")
        if value.position.state is PositionState.OPEN:
            messages.append("GUNCEL_RISK_DEGERLENDIRILEMIYOR")
        valid_context = False
    if value.missing_component:
        messages.append(value.missing_component)
        valid_context = False
    if not value.in_scope:
        messages.append("V1_DOGRULANMADI")
        valid_context = False
    if value.quantity_error:
        messages.append(value.quantity_error)
        valid_context = False
    pending = value.signal is Signal.BUY and not value.next_bar_available
    if pending:
        messages.append("SONRAKI_MUM_BEKLENIYOR")
        valid_context = False

    if valid_context:
        if value.signal is Signal.BUY:
            action = "SATIN AL" if value.position.state is PositionState.FLAT else "TUT"
        elif value.signal is Signal.SELL and value.position.state is PositionState.OPEN:
            action = "TAMAMINI SAT"
        elif value.signal is Signal.WAIT and value.position.state is PositionState.OPEN:
            action = "TUT"

    if value.ambiguous_sequence:
        messages.append("SIRA_BILINMIYOR")
    if value.suggested_stop is not None and value.position.state is PositionState.OPEN:
        messages.append("SABIT_STOP_SAT")
    if value.fee is None:
        messages.extend(("KOMISYON_BILINMIYOR", "NAKIT_YETERLILIGI_DOGRULANMADI"))
    if value.spread_bps is None:
        messages.append("MAKAS_BILINMIYOR")
    if value.slippage_bps is None:
        messages.append("KAYMA_BILINMIYOR")
    stop_verified = True
    if value.asset_kind in {"XAU", "XAU_GOLD"}:
        messages.append("GC=F VADELI ALTIN REFERANSI")
    elif value.asset_kind == "GRAM_TRY":
        messages.append("TURETILMIS_OHLC")
        stop_verified = False

    assumptions = tuple(
        label for label in (
            None if value.spread_bps is None else f"Makas: {_display_decimal(value.spread_bps)} bp",
            None if value.slippage_bps is None else f"Kayma: {_display_decimal(value.slippage_bps)} bp",
        ) if label is not None
    )
    provisional = None
    if value.fee is None and value.gross_result is not None:
        provisional = "GECICI_ZARAR" if value.gross_result < 0 else "GECICI_SONUC"
    return DecisionPanel(
        action, tuple(messages), value.missing_component is None,
        value.fee is not None, value.exit_reason,
        "Uyum puanı kazanma olasılığı değildir", assumptions,
        value.decision_at if value.data_status != "GECERLI" else None,
        getattr(value.position, "stop", None), "TEYITLI GERCEK", "SANAL STRATEJI",
        value.real_trade_confirmed, pending, provisional, stop_verified,
    )


def validate_fee(value):
    if value is None:
        return None
    amount = Decimal(value)
    return None if amount.is_finite() and Decimal("0") <= amount < Decimal("100") else "GECERSIZ_KOMISYON"


def validate_delay(value):
    return None if value is not None and value >= 0 else "GECERSIZ_TEPKI_SURESI"


def paper_defaults(currency):
    return Decimal("10000"), Decimal("1000"), currency


def aggregate_totals(values):
    output = {}
    usd = Decimal(values.get("USD", 0)) + Decimal(values.get("USDT", 0))
    if "USD" in values or "USDT" in values:
        output["USD/USDT"] = usd
    for currency, amount in values.items():
        if currency not in {"USD", "USDT"}:
            output[currency] = Decimal(amount)
    return output
