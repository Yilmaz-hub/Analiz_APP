from dataclasses import dataclass
from decimal import Decimal

from datetime import datetime, timezone

from trade_decisions import (
    Action, LOSS_COOLDOWN_BARS, PositionState, Signal, classify_exit, decide_action,
    validate_position,
)


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
    #: Güncel fiyat: açık pozisyonda stop temasını belirler (Q2). None = bilinmiyor.
    current_price: Decimal | None = None
    #: Zararlı çıkıştan beri kapanmış günlük mum sayısı (Q4). None = bekleme yok.
    bars_since_loss_exit: int | None = None


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


#: Ekranda ham makine kodu görünmesin (spec 0003, AK17 / Q12). Her kod için
#: Türkçe, eyleme dönük bir metin; bilinmeyen kod için güvenli genel metin.
CODE_TEXT = {
    "GECERLI": "Veri geçerli",
    "VERI_YOK": "Veri alınamadı",
    "GECERSIZ_VERI": "Veri geçersiz",
    "GECERSIZ_OHLC": "Fiyat verisi tutarsız (açılış/yüksek/düşük/kapanış)",
    "CELISKILI_TEKRAR": "Aynı mum için çelişkili kayıtlar var",
    "KARISIK_KAYNAK": "Fiyat geçmişi farklı kaynaklardan karışmış",
    "YETERSIZ_GECMIS": "Karar için yeterli fiyat geçmişi yok",
    "BILESEN_HAZIR_DEGIL": "Karar bileşeni hazır değil",
    "YENI_VERI_BEKLENIYOR": "Yeni günlük mumun yayımlanması bekleniyor",
    "ESKI_VERI": "Veri güncel değil",
    "ESKI_KARAR": "Gösterilen karar güncel değil",
    "ESKI_TEYITSIZ": "Teyitsiz eski karar",
    "V1_DOGRULANMADI": "Bu varlık/periyot için karar kuralları doğrulanmadı",
    "POZISYON_BILINMIYOR": "Pozisyon durumu bilinmiyor",
    "GECERSIZ_POZISYON": "Pozisyon kaydı geçersiz",
    "GUNCEL_RISK_DEGERLENDIRILEMIYOR": "Güncel risk değerlendirilemiyor",
    "SONRAKI_MUM_BEKLENIYOR": "İşlem, sonraki mumun açılışında yapılır; bekleniyor",
    "SIRA_BILINMIYOR": "Aynı mumda stop ile hedefin sırası bilinmiyor",
    "SABIT_STOP_SAT": "Stop sabit; önerilen seviye yalnız bilgidir",
    "KOMISYON_BILINMIYOR": "Komisyon bilinmiyor; sonuç brüt, net doğrulanmadı",
    "NAKIT_YETERLILIGI_DOGRULANMADI": "Nakit yeterliliği doğrulanmadı",
    "MAKAS_BILINMIYOR": "Alış-satış farkı (makas) bilinmiyor",
    "KAYMA_BILINMIYOR": "Kayma bilinmiyor",
    "MALIYET_BILINMIYOR": "İşlem maliyeti bilinmiyor",
    "GECERSIZ_MALIYET": "İşlem maliyeti geçersiz",
    "SONLU_OLMAYAN_MALIYET": "İşlem maliyeti sayı değil",
    "GECERSIZ_KOMISYON": "Komisyon geçersiz",
    "GECERSIZ_TEPKI_SURESI": "Tepki süresi geçersiz",
    "GECERSIZ_STOP": "Stop seviyesi geçersiz",
    "GECERSIZ_GERCEKLESME_FIYATI": "Gerçekleşme fiyatı geçersiz",
    "MIKTAR_ADIMI_BILINMIYOR": "Miktar adımı bilinmiyor; alım yapılmadı",
    "ASGARI_MIKTAR": "Miktar asgari işlem miktarının altında",
    "ASGARI_TUTAR": "Tutar asgari işlem tutarının altında",
    "YETERSIZ_NAKIT": "Nakit yetersiz",
    "ZORUNLU_ALAN_EKSIK": "Zorunlu alan eksik",
    "TURETILMIS_OHLC": "Fiyat serisi türetilmiş; stop simülasyonu doğrulanmadı",
    "BILESEN_YOK": "Karar bileşeni hesaplanamadı; yeni işlem sinyali yok",
    "GECERSIZ_MIKTAR": "Miktar sıfırdan büyük bir sayı olmalıdır",
    "GECERSIZ_FIYAT": "Fiyat sıfırdan büyük bir sayı olmalıdır",
    "GELECEK_ZAMAN": "İşlem zamanı gelecekte olamaz",
    "YETERSIZ_BAKIYE": "Bakiye yetersiz",
    "MIKTAR_FAZLA": "Satış miktarı eldeki miktardan fazla olamaz",
    "GECERSIZ_YUZDE": "Yüzde 0'dan büyük ve en fazla 100 olmalıdır",
    "YUZDE_SIFIRA_DUSTU": "Bu yüzde, ürünün miktar adımının altında kalıyor; daha yüksek bir yüzde ya da miktar girin",
    "CIKIS_GIRISTEN_ONCE": "Çıkış zamanı girişten önce olamaz",
    "TEKRAR_TEYIT": "Bu işlem daha önce teyit edilmiş; tekrar kaydedilmedi",
    "KAYIT_YAZILAMADI": "Kayıtlara şu anda yazılamadı; işlem kaydedilmedi",
    "GUNLUK_YAZILAMADI": "Pozisyon kaydedildi ancak işlem günlüğüne yazılamadı; aynı teyidi yeniden girerek tamamlayın",
    "STOP_TEMASI": "Güncel fiyat stop seviyesinde ya da altında; pozisyonun tamamı satılmalı",
    "BEKLEME": "Zararlı çıkıştan sonra bekleme sürüyor; yeni alım şimdilik gösterilmez",
    "GC=F VADELI ALTIN REFERANSI": "Altın fiyatı GC=F vadeli kontrat referansıdır",
}

COMPONENT_TEXT = {
    "trend": "trend", "momentum": "momentum", "volatility": "volatilite",
    "volume": "hacim", "regime": "rejim filtresi", "pattern": "formasyon",
    "ml": "yapay zekâ (ML)", "advanced": "gelişmiş analiz",
}

GENERIC_CODE_TEXT = "Durum doğrulanamadı"


def describe_code(code, components=()):
    """Makine kodunu Türkçe metne çevirir; bileşen adları varsa sona eklenir."""
    text = CODE_TEXT.get(str(code), GENERIC_CODE_TEXT)
    if components:
        names = ", ".join(COMPONENT_TEXT.get(c, str(c)) for c in components)
        text = f"{text}: {names}"
    return text


def resolve_decision_time(store, key, data_status, evaluated_at):
    """Karar zamanı yalnız veri geçerliyken ilerler (spec 0003, Q11).

    Veri geçerliyse `evaluated_at` kaydedilir ve döner. Değilse ekrana "şimdi"
    basılmaz: son geçerli kararın zamanı (yoksa None) döner.
    """
    if data_status == "GECERLI":
        store[key] = evaluated_at
        return evaluated_at
    return store.get(key)


def format_decision_time(moment):
    """Karar zamanını Türkiye saatinde gösterir."""
    from zoneinfo import ZoneInfo
    return moment.astimezone(ZoneInfo("Europe/Istanbul")).strftime("%d.%m.%Y %H:%M")


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

    exit_reason = value.exit_reason
    if valid_context:
        is_open = value.position.state is PositionState.OPEN
        stop = value.position.stop if is_open else None
        # Güncel fiyatı bilinmeyen açık pozisyonda "TUT" denemez (Q2); açık SAT sinyali yine geçerlidir.
        risk_unknown = is_open and stop is not None and value.current_price is None
        stop_touched = (is_open and stop is not None and value.current_price is not None
                        and value.current_price <= stop)
        cooldown = LOSS_COOLDOWN_BARS if value.bars_since_loss_exit is None else value.bars_since_loss_exit
        decision = decide_action(value.signal, value.position,
                                 stop_touched=stop_touched, cooldown_bars=cooldown)
        if risk_unknown and value.signal is not Signal.SELL:
            messages.append("GUNCEL_RISK_DEGERLENDIRILEMIYOR")
        elif decision.action is Action.SELL:
            action = "TAMAMINI SAT"
            if decision.reason == "STOP":
                messages.append("STOP_TEMASI")
                exit_reason = "STOP"
        elif decision.action is Action.HOLD:
            action = "TUT"
        elif decision.action is Action.BUY:
            action = "SATIN AL"
        elif value.signal is Signal.BUY:
            messages.append("BEKLEME")

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
        action, tuple(dict.fromkeys(messages)), value.missing_component is None,
        value.fee is not None, exit_reason,
        "Uyum puanı kazanma olasılığı değildir", assumptions,
        value.decision_at if value.data_status != "GECERLI" else None,
        getattr(value.position, "stop", None), "TEYITLI GERCEK", "SANAL STRATEJI",
        value.real_trade_confirmed, pending, provisional, stop_verified,
    )


def bars_since_loss_exit(positions, coin, daily_index, now):
    """Son gerçek çıkış zararlıysa, çıkış gününden beri kapanmış günlük mum sayısı.

    Zararsız (kâr ya da tam sıfır) çıkışta ya da çıkış yoksa None: bekleme yok.
    Komisyon kayıtlı olmadığından sınıflama geçicidir (`classify_exit`, AC86).
    Çıkış günü ve henüz kapanmamış bugünkü mum sayılmaz.
    """
    exits = [p for p in positions
             if p.get("Coin") == coin and str(p.get("Status", "")).startswith("CLOSED")
             and p.get("Çıkış Zamanı")]
    if not exits:
        return None
    latest = max(exits, key=lambda p: datetime.fromisoformat(p["Çıkış Zamanı"]))
    outcome = classify_exit(Decimal(str(latest.get("Realized", 0))), fee_known=False)
    if not outcome.cooldown_required:
        return None
    exit_day = datetime.fromisoformat(latest["Çıkış Zamanı"]).astimezone(timezone.utc).date()
    today = now.astimezone(timezone.utc).date()
    return sum(1 for stamp in daily_index if exit_day < stamp.date() < today)


def validate_fee(value):
    if value is None:
        return None
    amount = Decimal(value)
    return None if amount.is_finite() and Decimal("0") <= amount < Decimal("100") else "GECERSIZ_KOMISYON"


def validate_delay(value):
    return None if value is not None and value >= 0 else "GECERSIZ_TEPKI_SURESI"


def paper_defaults(currency):
    return Decimal("10000"), Decimal("1000"), currency
