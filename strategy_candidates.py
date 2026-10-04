"""Aday stratejiler: ön kayıt, parmak izi, Q03 ölçütü, geçmiş ve aktif strateji
(spec 0005, Adım 5 / S3).

* Aday; giriş/çıkış kuralı, piyasa, ayarlar, ölçüt sürümü ve değerlendirme
  dönemiyle **sonuç görülmeden** kaydedilir (R07). Ön kaydı olmayan adayla koşum
  başlamaz (AC69).
* Parmak izi, adı dışındaki tüm tanımın kanonik JSON'unun SHA-256'sıdır; bir ayar
  değişirse yeni sürümdür ve eski sürümün gözlem sayacı devralınmaz (AC79).
* Q03: getiri **kesin büyük**, maks. düşüş **büyük değil**, beklenti **kesin
  pozitif**, en az 30 kapanmış işlem. Sonuç üç değerlidir; kuralı onaysız aday
  yalnız "deneme"dir (AC48).
* Aktif strateji varsayılan V1'dir; değerlendirme sonucu onu değiştirmez, yalnız
  kullanıcının açık seçimi değiştirir (R09, AC01, AC17).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Mapping

import storage
from regime_classifier import REGIME_VERSION, YUKSELEN

REFERENCE_STRATEGY = "V1"
CRITERIA_VERSION = "Q03-1"
MIN_CLOSED_TRADES = 30

APPROVED_ENTRY_RULES = frozenset({"V1", "KIRILIM", "GERI_CEKILME"})
APPROVED_EXIT_RULES = frozenset({"V1"})

OLCUTU_KARSILADI = "OLCUTU_KARSILADI"
OLCUTU_KARSILAMADI = "OLCUTU_KARSILAMADI"
YETERSIZ_VERI = "YETERSIZ_VERI"
DENEME = "DENEME"

_REGISTRY_KEY = "strategy_candidates"
_HISTORY_KEY = "candidate_history"
_ACTIVE_KEY = "active_strategy"


@dataclass(frozen=True)
class Candidate:
    name: str
    entry_rule: str
    exit_rule: str
    market: str
    settings: Mapping[str, str] = field(default_factory=dict)
    period_start: date | None = None
    period_end: date | None = None
    criteria: str = CRITERIA_VERSION
    regime_version: str = REGIME_VERSION


@dataclass(frozen=True)
class Metrics:
    net_return_pct: Decimal | None
    max_drawdown_pct: Decimal | None
    expectancy: Decimal | None
    closed_count: int


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str = ""


@dataclass(frozen=True)
class RunPermit:
    ok: bool
    reason: str = ""


def _definition(candidate: Candidate) -> dict:
    body = asdict(candidate)
    body.pop("name")
    body["settings"] = dict(sorted((str(k), str(v)) for k, v in candidate.settings.items()))
    for key in ("period_start", "period_end"):
        body[key] = None if body[key] is None else body[key].isoformat()
    return body


def fingerprint(candidate: Candidate) -> str:
    canonical = json.dumps(_definition(candidate), sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_approved(candidate: Candidate) -> bool:
    return (candidate.entry_rule in APPROVED_ENTRY_RULES
            and candidate.exit_rule in APPROVED_EXIT_RULES)


def _read(key: str, default):
    value = storage.read_doc(key)
    return default if value is None else value


def preregister(candidate: Candidate, registered_at: datetime) -> str:
    """Adayı sonuç görülmeden kaydeder; parmak izini döndürür. Var olan kayıt değişmez."""
    key = fingerprint(candidate)
    registry = _read(_REGISTRY_KEY, {})
    if key not in registry:
        registry[key] = {"name": candidate.name, "definition": _definition(candidate),
                         "registered_at": registered_at.isoformat()}
        storage.write_doc(_REGISTRY_KEY, registry)
    return key


def start_run(candidate: Candidate) -> RunPermit:
    if fingerprint(candidate) not in _read(_REGISTRY_KEY, {}):
        return RunPermit(False, "Aday, ölçütleri ve değerlendirme dönemiyle önceden kaydedilmemiş; "
                                "koşum başlatılmadı.")
    return RunPermit(True)


def judge(candidate: Metrics, reference: Metrics, approved: bool = True) -> Verdict:
    if not approved:
        return Verdict(DENEME, "Giriş veya çıkış kuralı onaylı değil; sonuç yalnız denemedir.")
    if candidate.closed_count < MIN_CLOSED_TRADES:
        return Verdict(YETERSIZ_VERI, f"Kapanmış işlem {candidate.closed_count}; en az "
                                      f"{MIN_CLOSED_TRADES} gerekir.")
    values = (candidate.net_return_pct, candidate.max_drawdown_pct, candidate.expectancy,
              reference.net_return_pct, reference.max_drawdown_pct)
    if any(value is None for value in values):
        return Verdict(YETERSIZ_VERI, "Ölçütlerden biri hesaplanamıyor.")
    failures = []
    if not candidate.net_return_pct > reference.net_return_pct:
        failures.append("getiri referanstan yüksek değil")
    if candidate.max_drawdown_pct > reference.max_drawdown_pct:
        failures.append("maksimum düşüş referanstan büyük")
    if not candidate.expectancy > 0:
        failures.append("işlem beklentisi pozitif değil")
    if failures:
        return Verdict(OLCUTU_KARSILAMADI, "; ".join(failures).capitalize() + ".")
    return Verdict(OLCUTU_KARSILADI)


def judge_by_market(pairs: Mapping[str, tuple[Metrics, Metrics]],
                    approved: bool = True) -> dict[str, Verdict]:
    """Her piyasa kendi verisiyle ayrı değerlendirilir; başarı aktarılmaz (R08, AC22)."""
    return {market: judge(cand, ref, approved) for market, (cand, ref) in pairs.items()}


def record_result(candidate: Candidate, market: str, verdict: Verdict,
                  recorded_at: datetime) -> None:
    """Sonucu değerlendirme geçmişine ekler; başarısız sonuç da silinmez (R07, AC16)."""
    entries = _read(_HISTORY_KEY, [])
    entries.append({"fingerprint": fingerprint(candidate), "name": candidate.name,
                    "market": market, "status": verdict.status, "reason": verdict.reason,
                    "recorded_at": recorded_at.isoformat()})
    storage.write_doc(_HISTORY_KEY, entries)


def history() -> list[dict]:
    return list(_read(_HISTORY_KEY, []))


def observation_count(candidate: Candidate) -> int:
    key = fingerprint(candidate)
    return sum(1 for entry in history() if entry["fingerprint"] == key)


def active_strategy() -> str:
    return _read(_ACTIVE_KEY, {}).get("strategy", REFERENCE_STRATEGY)


def choose_active_strategy(candidate: Candidate | None) -> None:
    """Yalnız kullanıcının açık seçimiyle çağrılır; `None` mevcut V1'e döner."""
    value = REFERENCE_STRATEGY if candidate is None else fingerprint(candidate)
    storage.write_doc(_ACTIVE_KEY, {"strategy": value})


def single_filter_effect(first: Candidate, second: Candidate) -> str | None:
    """Yalnız tek ayarı farklı iki aday için o ayarın adı; aksi halde `None` (AC15)."""
    a, b = _definition(first), _definition(second)
    if {k: v for k, v in a.items() if k != "settings"} != {k: v for k, v in b.items() if k != "settings"}:
        return None
    keys = set(a["settings"]) | set(b["settings"])
    differing = [k for k in sorted(keys) if a["settings"].get(k) != b["settings"].get(k)]
    return differing[0] if len(differing) == 1 else None


def default_candidates(market: str, period_start: date | None,
                       period_end: date | None) -> list[Candidate]:
    """Q02 karar kaydındaki iki giriş adayı; çıkış V1 ile aynıdır."""
    return [
        Candidate("Kırılım (20 gün)", "KIRILIM", "V1", market, {"lookback": "20"},
                  period_start, period_end),
        Candidate("Geri çekilme (EMA20, %1)", "GERI_CEKILME", "V1", market,
                  {"ema": "20", "tolerance_pct": "1"}, period_start, period_end),
    ]


def registered_name(key: str) -> str | None:
    entry = _read(_REGISTRY_KEY, {}).get(key)
    return None if entry is None else entry["name"]


def metrics_from_backtest(backtest: dict) -> Metrics:
    """V1 backtest çıktısından Q03 ölçütleri (günlük kapanış sermayesi + kapanmış işlemler)."""
    from performance_report import EquityPoint, max_drawdown, net_return

    points = [EquityPoint(_day(p["date"]), Decimal(p["equity"])) for p in backtest["daily_equity"]]
    pnls = [Decimal(t["pnl"]) for t in backtest["trades"]]
    return Metrics(
        net_return_pct=net_return(points),
        max_drawdown_pct=max_drawdown(points),
        expectancy=(sum(pnls, Decimal("0")) / len(pnls)) if pnls else None,
        closed_count=len(pnls),
    )


def _day(value) -> date:
    return value.date() if hasattr(value, "date") else value


def candidate_decisions(frame, v1_decisions: Mapping, candidate: Candidate, regimes, *,
                        lookback: int = 20, ema_span: int = 20,
                        tolerance_pct: Decimal = Decimal("1")) -> dict:
    """Aday giriş kararları; çıkış V1 ile aynıdır (V1'in SAT kararları korunur, R06).

    Girişler yalnız yükselen piyasa gününde üretilir (Q02). Her gün yalnız o güne
    kadarki veriyi kullanır.
    """
    close = frame["Close"]
    ema = close.ewm(span=ema_span, adjust=False).mean()
    factor = Decimal("1") + Decimal(tolerance_pct) / 100
    decisions = {}
    for position, day in enumerate(frame.index):
        if v1_decisions.get(day) == "SAT":
            decisions[day] = "SAT"
            continue
        entry = False
        if regimes.iloc[position] == YUKSELEN:
            today = Decimal(str(close.iloc[position]))
            if candidate.entry_rule == "KIRILIM" and position >= lookback:
                prior = close.iloc[position - lookback:position].max()
                entry = today > Decimal(str(prior))
            elif candidate.entry_rule == "GERI_CEKILME":
                average = Decimal(str(ema.iloc[position]))
                low = Decimal(str(frame["Low"].iloc[position]))
                entry = low <= average * factor and today > average
            elif candidate.entry_rule == "V1":
                entry = v1_decisions.get(day) == "AL"
        decisions[day] = "AL" if entry else "BEKLE"
    return decisions
