"""İşlem varsayımları (spec 0003, Q7): sermaye, işlem tutarı, miktar adımı, makas,
kayma ve komisyon.

Geçmiş test ve sanal takip aynı kayıttan okur; böylece iki ekran farklı varsayımla
çalışamaz. Değerler kalıcı depoda saklanır (`docs/storage.md`). Kurallar:

* Boş alan = **bilinmiyor** (`None`); `0` = **açıkça sıfır**. Bilinmeyen maliyet
  sıfır maliyet diye sunulmaz.
* Negatif ya da sayı olmayan değer, değerlendirme başlamadan Türkçe mesajla reddedilir.
* Varsayımlar varlık başına tutulur; bir varlığın ayarı başkasını değiştirmez (AC98).
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import storage
from trade_execution import CostAssumptions, TradeSettings

DEFAULT_CAPITAL = "10000"      # AC90
DEFAULT_NOTIONAL = "1000"      # AC90

FIELDS = ("capital", "notional", "quantity_step", "spread_bps", "slippage_bps", "commission_pct")


@dataclass(frozen=True)
class ParsedSettings:
    capital: Decimal | None
    settings: TradeSettings | None
    errors: tuple

    @property
    def ok(self):
        return not self.errors


def _number(text):
    """Boş = None; aksi halde Decimal. Sayı olmayan değerde ValueError."""
    if text is None:
        return None
    value = str(text).strip().replace(",", ".")
    if value == "":
        return None
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(value) from exc
    if not number.is_finite():
        raise ValueError(value)
    return number


def parse_settings(raw):
    """Ham alanları doğrular; hata varsa değerlendirme başlatılmaz (AC26)."""
    raw = raw or {}
    labels = {
        "capital": "Başlangıç sermayesi", "notional": "İşlem tutarı",
        "quantity_step": "Adet/lot adımı", "spread_bps": "Makas (baz puan)",
        "slippage_bps": "Kayma (baz puan)", "commission_pct": "Komisyon (%)",
    }
    values, errors = {}, []
    for name in FIELDS:
        try:
            values[name] = _number(raw.get(name))
        except ValueError:
            errors.append(f"{labels[name]} sayı olmalıdır.")
            values[name] = None
    if errors:
        return ParsedSettings(None, None, tuple(errors))

    capital = values["capital"] if values["capital"] is not None else Decimal(DEFAULT_CAPITAL)
    notional = values["notional"] if values["notional"] is not None else Decimal(DEFAULT_NOTIONAL)
    if capital <= 0:
        errors.append("Başlangıç sermayesi sıfırdan büyük olmalıdır.")
    if notional <= 0:
        errors.append("İşlem tutarı sıfırdan büyük olmalıdır.")
    if values["quantity_step"] is not None and values["quantity_step"] <= 0:
        errors.append("Adet/lot adımı sıfırdan büyük olmalıdır; bilinmiyorsa boş bırakın.")
    for name in ("spread_bps", "slippage_bps"):
        if values[name] is not None and values[name] < 0:
            errors.append(f"{labels[name]} negatif olamaz.")
    commission = values["commission_pct"]
    if commission is not None and not (Decimal("0") <= commission < Decimal("100")):
        errors.append("Komisyon negatif olamaz ve %100'den küçük olmalıdır.")   # AC26 / AC113
    if errors:
        return ParsedSettings(None, None, tuple(errors))

    costs = CostAssumptions(values["spread_bps"], values["slippage_bps"], commission)
    return ParsedSettings(capital, TradeSettings(notional, values["quantity_step"], costs), ())


def to_raw(values):
    """Depoya yazılacak alanlar: yalnız bilinen alanlar metin olarak."""
    return {name: ("" if values.get(name) is None else str(values[name]).strip()) for name in FIELDS}


def load_raw(asset):
    """Varlığın kayıtlı ham varsayımları; kayıt yoksa boş. Erişilemezse StorageAccessError."""
    document = storage.read_doc(storage.TRADE_SETTINGS_KEY) or {}
    return dict(document.get(asset, {}))


def known_quantity_step(asset):
    """Varlığın kayıtlı miktar adımı; bilinmiyor ya da ayarlar geçersizse None."""
    parsed = parse_settings(load_raw(asset))
    return parsed.settings.quantity_step if parsed.ok else None


def save_raw(asset, values):
    """Varlığın varsayımlarını yazar; diğer varlıkların kaydına dokunmaz (AC98)."""
    document = storage.read_doc(storage.TRADE_SETTINGS_KEY) or {}
    document[asset] = to_raw(values)
    storage.write_doc(storage.TRADE_SETTINGS_KEY, document)


def step_cost(settings, price):
    """Bir adım miktarın güncel fiyatla tutarı; adım ya da fiyat bilinmiyorsa None (spec 0013)."""
    if settings is None or settings.quantity_step is None or not price:
        return None
    try:
        value = Decimal(str(price))
    except InvalidOperation:
        return None
    if not value.is_finite() or value <= 0:      # NaN/inf (bozuk son mum) uyarı üretmez, koşucuyu düşürmez
        return None
    return settings.quantity_step * value


def step_exceeds_notional(settings, price):
    """Tek adım bile işlem tutarını aşıyorsa tutarı döner (hiç alım yapılamaz); aksi halde None."""
    cost = step_cost(settings, price)
    return cost if cost is not None and cost > settings.notional else None
