"""İleri dönem sanal takip kaydı (spec 0005, Adım 7 / S6).

Kararlar `app_records` belge deposuna değil, ayrı tablolara yazılır (plan D1):
tekillik anahtarı (varlık, strateji sürümü, dayanak mum günü) **veritabanında**
zorlanır; eşzamanlı ikinci yazım hata değil yok sayma üretir (R17, AC70).

* Bir karar, verinin kullanılabilir olduğu andan sonra `ON_TIME_HOURS` içinde
  kaydedildiyse "zamanında"dır; aksi halde "sonradan oluşturuldu" (R16, AC35, AC38).
* Mum sonradan değişirse karar dondurulmuş kalır, revizyon ayrı kaydedilir (AC71).
* Yeterlilik sayacı yalnız zamanında ve gerçek saatle üretilmiş gözlemleri sayar;
  sürüm başına ayrıdır (AC42, AC63, AC87). Gerçek işlemler `position_journal`'dadır
  ve bu sayaçları değiştirmez (R19, AC43, AC76).
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Iterable, Mapping

import market_map
import storage
from config import ForwardConfig
from logger import logger
from regime_classifier import DUSEN, YATAY, YUKSELEN

REFERENCE_VERSION = "V1"
LATE_FLAG = "geç kayıtla güncellendi"
_REGIME_NAMES = {YUKSELEN: "Yükselen", DUSEN: "Düşen", YATAY: "Yatay"}

_DDL = (
    "CREATE TABLE IF NOT EXISTS forward_decisions ("
    "asset VARCHAR(64) NOT NULL, strategy_version VARCHAR(96) NOT NULL, "
    "candle_day VARCHAR(10) NOT NULL, decision VARCHAR(16) NOT NULL, "
    "source VARCHAR(64) NOT NULL, evaluated_at VARCHAR(40) NOT NULL, candle TEXT NOT NULL, "
    "assumptions TEXT NOT NULL, on_time INTEGER NOT NULL, real_clock INTEGER NOT NULL, "
    "regime VARCHAR(16), regime_version VARCHAR(16), "
    "PRIMARY KEY (asset, strategy_version, candle_day))",
    "CREATE TABLE IF NOT EXISTS forward_revisions ("
    "asset VARCHAR(64) NOT NULL, strategy_version VARCHAR(96) NOT NULL, "
    "candle_day VARCHAR(10) NOT NULL, new_candle TEXT NOT NULL, "
    "PRIMARY KEY (asset, strategy_version, candle_day, new_candle))",
    "CREATE TABLE IF NOT EXISTS forward_trades ("
    "asset VARCHAR(64) NOT NULL, strategy_version VARCHAR(96) NOT NULL, "
    "entry_day VARCHAR(10) NOT NULL, exit_day VARCHAR(10) NOT NULL, pnl VARCHAR(40) NOT NULL, "
    "PRIMARY KEY (asset, strategy_version, entry_day))",
    "CREATE TABLE IF NOT EXISTS forward_runs ("
    "run_at VARCHAR(40) PRIMARY KEY, ok INTEGER NOT NULL, note TEXT NOT NULL)",
)


@dataclass(frozen=True)
class ForwardDecision:
    asset: str
    strategy_version: str
    candle_day: date
    decision: str
    source: str
    evaluated_at: datetime
    candle: Mapping[str, str]
    assumptions: Mapping[str, str] = field(default_factory=dict)
    on_time: bool = True
    real_clock: bool = True
    regime: str | None = None
    regime_version: str | None = None


@dataclass(frozen=True)
class Sufficiency:
    sufficient: bool
    missing: list[str]


def _guard(function):
    """Veritabanı/dosya hatalarını `StorageAccessError`'a çevirir; ham SQL kullanıcıya sızmaz (AC100)."""
    from sqlalchemy.exc import SQLAlchemyError

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except storage.StorageAccessError:
            raise
        except (SQLAlchemyError, OSError) as exc:
            logger.error(f"Forward tracker storage error ({function.__name__}): {exc}")
            raise storage.StorageAccessError("İleri takip kayıtlarına erişilemedi.") from exc

    return wrapper


def _engine():
    engine = storage.get_engine()
    ensure_tables(engine)
    return engine


@_guard
def ensure_tables(engine=None) -> None:
    from sqlalchemy import text

    engine = engine or storage.get_engine()
    with engine.begin() as conn:
        for statement in _DDL:
            conn.execute(text(statement))


def available_at(market: str, candle_day: date) -> datetime:
    """Günlük mumun kullanılabilir olduğu an (UTC)."""
    hour, minute = ForwardConfig.SESSION_CLOSE_UTC[market]
    base = datetime.combine(candle_day, time(0, 0), tzinfo=timezone.utc)
    return base + timedelta(hours=hour, minutes=minute)


def is_on_time(market: str, candle_day: date, evaluated_at: datetime) -> bool:
    deadline = available_at(market, candle_day) + timedelta(hours=ForwardConfig.ON_TIME_HOURS)
    return evaluated_at <= deadline


@_guard
def record(decision: ForwardDecision) -> bool:
    """Kararı yazar; kayıt zaten varsa yok sayar (revizyonu ayrıca işaretler). Yazıldıysa True."""
    from sqlalchemy import text

    row = {
        "asset": decision.asset, "version": decision.strategy_version,
        "day": decision.candle_day.isoformat(), "decision": decision.decision,
        "source": decision.source, "evaluated": decision.evaluated_at.isoformat(),
        "candle": json.dumps(dict(decision.candle), sort_keys=True),
        "assumptions": json.dumps(dict(decision.assumptions), sort_keys=True, ensure_ascii=False),
        "on_time": int(decision.on_time), "real_clock": int(decision.real_clock),
        "regime": decision.regime, "regime_version": decision.regime_version,
    }
    with _engine().begin() as conn:
        inserted = conn.execute(text(
            "INSERT INTO forward_decisions (asset, strategy_version, candle_day, decision, source, "
            "evaluated_at, candle, assumptions, on_time, real_clock, regime, regime_version) "
            "VALUES (:asset, :version, :day, :decision, :source, :evaluated, :candle, :assumptions, "
            ":on_time, :real_clock, :regime, :regime_version) ON CONFLICT DO NOTHING"), row).rowcount
        if inserted == 1:
            return True
        stored = conn.execute(text(
            "SELECT candle FROM forward_decisions WHERE asset = :asset AND strategy_version = :version "
            "AND candle_day = :day"), row).scalar_one()
        if stored != row["candle"]:
            conn.execute(text(
                "INSERT INTO forward_revisions (asset, strategy_version, candle_day, new_candle) "
                "VALUES (:asset, :version, :day, :candle) ON CONFLICT DO NOTHING"), row)
    return False


def _from_row(row) -> ForwardDecision:
    return ForwardDecision(
        asset=row.asset, strategy_version=row.strategy_version,
        candle_day=date.fromisoformat(row.candle_day), decision=row.decision, source=row.source,
        evaluated_at=datetime.fromisoformat(row.evaluated_at), candle=json.loads(row.candle),
        assumptions=json.loads(row.assumptions), on_time=bool(row.on_time),
        real_clock=bool(row.real_clock), regime=row.regime, regime_version=row.regime_version)


@_guard
def decisions(asset: str, version: str) -> list[ForwardDecision]:
    from sqlalchemy import text

    with _engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM forward_decisions WHERE asset = :a AND strategy_version = :v "
            "ORDER BY candle_day"), {"a": asset, "v": version}).fetchall()
    return [_from_row(row) for row in rows]


@_guard
def versions(asset: str) -> list[str]:
    """Varlık için kayıtlı strateji sürümü etiketleri."""
    from sqlalchemy import text

    with _engine().connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT strategy_version FROM forward_decisions "
                                 "WHERE asset = :a ORDER BY strategy_version"), {"a": asset}).fetchall()
    return [row[0] for row in rows]


@_guard
def tracked_pairs() -> list[tuple[str, str]]:
    from sqlalchemy import text

    with _engine().connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT asset, strategy_version FROM forward_decisions "
                                 "ORDER BY asset, strategy_version")).fetchall()
    return [(row.asset, row.strategy_version) for row in rows]


def get_decision(asset: str, version: str, candle_day: date) -> ForwardDecision | None:
    return next((d for d in decisions(asset, version) if d.candle_day == candle_day), None)


def decision_count(asset: str, version: str) -> int:
    return len(decisions(asset, version))


def status_text(decision: ForwardDecision) -> str:
    if not decision.real_clock:
        return "hızlandırılmış test (sayılmaz)"
    return "zamanında" if decision.on_time else "sonradan oluşturuldu"


@_guard
def revisions(asset: str, version: str) -> list[dict]:
    from sqlalchemy import text

    with _engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT candle_day, new_candle FROM forward_revisions WHERE asset = :a AND "
            "strategy_version = :v ORDER BY candle_day"), {"a": asset, "v": version}).fetchall()
    return [{"candle_day": row.candle_day, "new_candle": json.loads(row.new_candle)} for row in rows]


def _counted(asset: str, version: str) -> list[ForwardDecision]:
    return [d for d in decisions(asset, version) if d.on_time and d.real_clock]


def tracked_days(asset: str, version: str) -> int:
    return len(_counted(asset, version))


def regimes_seen(asset: str, version: str) -> set[str]:
    return {d.regime for d in _counted(asset, version) if d.regime}


@_guard
def save_trades(asset: str, version: str, trades: Iterable[Mapping]) -> None:
    from sqlalchemy import text

    with _engine().begin() as conn:
        for trade in trades:
            conn.execute(text(
                "INSERT INTO forward_trades (asset, strategy_version, entry_day, exit_day, pnl) "
                "VALUES (:a, :v, :entry, :exit, :pnl) ON CONFLICT DO NOTHING"),
                {"a": asset, "v": version, "entry": trade["entry_day"].isoformat(),
                 "exit": trade["exit_day"].isoformat(), "pnl": str(trade["pnl"])})


@_guard
def closed_trades(asset: str, version: str) -> int:
    """Yalnız zamanında ve gerçek saatle üretilmiş kararlardan oluşan sanal işlemleri sayar.

    Bir işlemin kararları: girişe karar verilen gün (girişten önceki son kayıtlı gün) ile
    çıkış günü arasındaki tüm kayıtlı kararlar. Hepsi sayılan karar değilse işlem sayılmaz
    (geç oluşturulan ya da hızlandırılmış gözlem yeterlilik sayacına girmez; AC87, AC92)."""
    from sqlalchemy import text

    stored = decisions(asset, version)
    all_days = [d.candle_day for d in stored]
    counted = {d.candle_day for d in stored if d.on_time and d.real_clock}
    with _engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT entry_day, exit_day FROM forward_trades WHERE asset = :a AND strategy_version = :v"),
            {"a": asset, "v": version}).fetchall()
    total = 0
    for entry_text, exit_text in rows:
        entry, exit_ = date.fromisoformat(entry_text), date.fromisoformat(exit_text)
        earlier = [day for day in all_days if day < entry]
        start_day = earlier[-1] if earlier else entry
        window = [day for day in all_days if start_day <= day <= exit_]
        if window and all(day in counted for day in window):
            total += 1
    return total


def sufficiency(days: int, trades: int, regimes: set[str]) -> Sufficiency:
    """Q07: ≥ 90 zamanında izlenen gün, ≥ 30 kapanmış işlem, üç piyasa koşulu birlikte."""
    missing = []
    if days < ForwardConfig.MIN_TRACKED_DAYS:
        missing.append(f"izlenen gün {days}/{ForwardConfig.MIN_TRACKED_DAYS}")
    if trades < ForwardConfig.MIN_CLOSED_TRADES:
        missing.append(f"kapanmış işlem {trades}/{ForwardConfig.MIN_CLOSED_TRADES}")
    for regime in (YUKSELEN, DUSEN, YATAY):
        if regime not in regimes:
            missing.append(f"{_REGIME_NAMES[regime]} piyasa gözlemi yok")
    return Sufficiency(not missing, missing)


def missing_days(asset: str, version: str) -> list[date]:
    """İlk ve son kayıtlı gün arasında kaydı olmayan günler (kripto her gün, diğerleri hafta içi).

    Resmi tatiller eksik görünebilir; bu bir uyarıdır, kesin hüküm değildir."""
    days = sorted(d.candle_day for d in decisions(asset, version))
    if len(days) < 2:
        return []
    info = market_map.market_of(asset)
    every_day = info is not None and info[0] == "KRIPTO"
    present, missing, cursor = set(days), [], days[0]
    while cursor <= days[-1]:
        if cursor not in present and (every_day or cursor.weekday() < 5):
            missing.append(cursor)
        cursor += timedelta(days=1)
    return missing


@_guard
def recent_runs(limit: int = 3) -> list[dict]:
    from sqlalchemy import text

    with _engine().connect() as conn:
        rows = conn.execute(text("SELECT run_at, ok, note FROM forward_runs ORDER BY run_at DESC LIMIT :n"),
                            {"n": limit}).fetchall()
    return [{"run_at": datetime.fromisoformat(r.run_at), "ok": bool(r.ok), "note": r.note} for r in rows]


def assess(asset: str, version: str) -> Sufficiency:
    return sufficiency(tracked_days(asset, version), closed_trades(asset, version),
                       regimes_seen(asset, version))


@_guard
def record_run(run_at: datetime, ok: bool, note: str) -> None:
    from sqlalchemy import text

    with _engine().begin() as conn:
        conn.execute(text("INSERT INTO forward_runs (run_at, ok, note) VALUES (:r, :o, :n) "
                          "ON CONFLICT DO NOTHING"),
                     {"r": run_at.isoformat(), "o": int(ok), "n": note})


@_guard
def last_successful_run() -> datetime | None:
    from sqlalchemy import text

    with _engine().connect() as conn:
        value = conn.execute(text("SELECT MAX(run_at) FROM forward_runs WHERE ok = 1")).scalar_one()
    return None if value is None else datetime.fromisoformat(value)


def period_flag(real_trades: Iterable[Mapping], start: date, end: date) -> str:
    """Kapanmış döneme sonradan kaydedilen gerçek işlem varsa dönem işareti (Q08, AC76)."""
    for trade in real_trades:
        executed, recorded = trade["executed_at"].date(), trade["recorded_at"].date()
        if start <= executed <= end and recorded > end:
            return LATE_FLAG
    return ""


def managing_version(position: Mapping, active: str) -> str:
    """Açık gerçek pozisyon, girişindeki sürümle yönetilir; aktif sürüm yalnız yeni girişlere (Q08)."""
    return position.get("StrategyVersion", REFERENCE_VERSION)
