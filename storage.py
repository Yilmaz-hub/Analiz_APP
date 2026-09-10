"""Kalıcı kayıt deposu (spec 0004).

Varlık listesi ve portföy kayıtları, bu modüldeki tek arayüz üzerinden okunup
yazılır. Amaç spec 0004'ün iki kayıp mekanizmasını da kapatmaktır:

1. **Çalışma dizinine bağlılık (P4/AC18).** Eski sürüm kayıtları `portfolio.json`
   gibi *göreli* adlarla yazıyordu; süreç hangi klasörden başlatıldıysa kayıt
   oraya düşüyor, başka bir klasörden açılınca başka bir kayıt kümesi
   görünüyordu. Depo artık ortamdan bağımsız tek bir yeri hedefler.
2. **Yayın ortamının diskinin kalıcı olmaması (P7/AC19).** Kayıtlar sürüm
   kontrolünde izlenmediği için yayın her kurulduğunda yerleşik varsayılanlara
   dönülüyordu. Bu yüzden depo, uygulama sürecinin dışında bir veri kaynağına
   (yönetilen Postgres) yönlendirilebilir olmalıdır.

Tek kod yolu, iki arka uç — ikisi de SQLAlchemy URL'i ile ayrışır:

* **Yerel:** SQLite dosyası, `data_dir()/analiz.db`.
* **Yayın:** `ANALIZ_APP_DB_URL` ya da `st.secrets["db_url"]` ile verilen
  yönetilen veritabanı.

Kayıtlar bugünkü JSON sözlüğünün aynısı olarak, belge biçiminde saklanır
(`app_records` tablosu, tek satır = tek belge). Alan alan şemaya çevirme
yapılmaz; böylece mevcut para/yüzde gösterimi olduğu gibi korunur (G16).

**Hata kuralı (R8/R8.1):** Okuma başarısız olursa `StorageAccessError` yükselir.
Bu modül hiçbir koşulda varsayılan ya da boş veri döndürmez — sessizce
varsayılana düşmek, kaybın kendisiydi.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from exceptions import ProTraderError
from logger import logger

# Belge anahtarları — dosya adlarının yerini alır.
ASSETS_KEY = "assets"
PORTFOLIO_KEY = "portfolio"

DATA_DIR_ENV = "ANALIZ_APP_DATA_DIR"
DB_URL_ENV = "ANALIZ_APP_DB_URL"

_TABLE = "app_records"
_engine = None
_engine_url = None


class StorageAccessError(ProTraderError):
    """Kayıtlara erişilemedi.

    Çağıran taraf bu hatayı yakalayıp **varsayılan veriyle devam etmez**;
    kullanıcıya erişim sorununu gösterir ve kayıt değiştiren işlemleri
    engeller (spec 0004, R8.1 / AC16 / AC16b).
    """


def data_dir() -> Path:
    """Kayıtların tutulduğu, çalışma dizininden bağımsız klasör (AC18)."""
    configured = os.environ.get(DATA_DIR_ENV)
    base = Path(configured) if configured else Path.home() / ".analiz_app"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _secret_db_url():
    """`st.secrets["db_url"]` degeri - yoksa None.

    Secrets dosyası bulunmadığında Streamlit istisna yükseltir; bu normal bir
    durumdur (yerel geliştirme) ve sessizce yerel arka uca düşülür.
    """
    try:
        import streamlit as st

        return st.secrets["db_url"]  # type: ignore[index]
    except Exception:
        return None


def db_url() -> str:
    """Kullanılacak veritabanı URL'i. Ortam değişkeni > secrets > yerel SQLite."""
    return (
        os.environ.get(DB_URL_ENV)
        or _secret_db_url()
        or f"sqlite:///{(data_dir() / 'analiz.db').as_posix()}"
    )


def is_remote_backend() -> bool:
    """Kayıtlar uygulama sürecinin dışında bir veritabanında mı tutuluyor?

    Yayın ortamında yeniden başlatma/yeniden yayınlama kaybını (AC19) yalnız
    uzak arka uç önler; arayüz durum rozeti bunu kullanır.
    """
    return not db_url().startswith("sqlite")


def get_engine():
    """Tekil SQLAlchemy engine; tablo yoksa oluşturur."""
    global _engine, _engine_url
    url = db_url()
    if _engine is not None and _engine_url == url:
        return _engine
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover - bağımlılık eksikse
        raise StorageAccessError("Kayıt deposu sürücüsü yüklenemedi.") from exc
    try:
        engine = create_engine(url, future=True)
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
                    "doc_key VARCHAR(64) PRIMARY KEY, "
                    "payload TEXT NOT NULL, "
                    "updated_at VARCHAR(32) NOT NULL)"
                )
            )
    except StorageAccessError:
        raise
    except Exception as exc:
        logger.error(f"Storage engine error: {exc}")
        raise StorageAccessError("Kayıt deposuna bağlanılamadı.") from exc
    _engine, _engine_url = engine, url
    return engine


def reset_engine():
    """Engine önbelleğini bırakır (test ve ortam değişikliği için)."""
    global _engine, _engine_url
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Engine dispose failed: {exc}")
    _engine, _engine_url = None, None


def read_doc(key: str):
    """Belgeyi döndürür; kayıt yoksa None.

    Erişim ya da çözümleme hatasında `StorageAccessError` yükselir — asla
    varsayılan veri döndürülmez.
    """
    from sqlalchemy import text

    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT payload FROM {_TABLE} WHERE doc_key = :k"), {"k": key}
            ).fetchone()
    except Exception as exc:
        logger.error(f"Storage read error ({key}): {exc}")
        raise StorageAccessError("Kayıtlar okunamadı.") from exc
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception as exc:
        logger.error(f"Storage decode error ({key}): {exc}")
        raise StorageAccessError("Kayıtlar okunamadı.") from exc


def write_doc(key: str, payload) -> None:
    """Belgeyi tek satırlık upsert ile yazar.

    Tek satır güncellemesi veritabanı tarafından atomiktir; eski sürümdeki
    "geçici dosya + `os.replace`" deseninin karşılığıdır.
    """
    from sqlalchemy import text

    engine = get_engine()
    body = json.dumps(payload, ensure_ascii=False)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {_TABLE} (doc_key, payload, updated_at) "
                    "VALUES (:k, :p, :u) "
                    "ON CONFLICT (doc_key) DO UPDATE SET "
                    "payload = excluded.payload, updated_at = excluded.updated_at"
                ),
                {"k": key, "p": body, "u": stamp},
            )
    except Exception as exc:
        logger.error(f"Storage write error ({key}): {exc}")
        raise StorageAccessError("Kayıtlar yazılamadı.") from exc


def check_access() -> tuple[bool, str]:
    """(erişilebilir mi, kullanıcıya gösterilecek metin).

    Teknik hata metni kullanıcıya sızmaz (conventions: Hata Yönetimi).
    """
    try:
        read_doc(ASSETS_KEY)
    except StorageAccessError:
        return False, "Kayıtlara şu anda erişilemiyor; değişiklik yapılamaz."
    if is_remote_backend():
        return True, "Kayıtlar korunuyor."
    return True, "Kayıtlar bu cihazda korunuyor."


# --- Eski dosya kayıtlarının içeri alınması (AC01 / AC01b) --------------------

# Belge anahtarı -> eski dosya adı.
LEGACY_FILES = {
    ASSETS_KEY: "varliklar.json",
    PORTFOLIO_KEY: "portfolio.json",
}


def legacy_search_dirs() -> list[Path]:
    """Eski kayıtların aranacağı yerler: çalışma dizini, proje klasörü, masaüstü."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent, Path.home() / "Desktop"]
    seen, out = set(), []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def import_legacy_documents() -> list[str]:
    """Depoda karşılığı olmayan belgeleri eski dosyalardan **kopyalayarak** alır.

    Kaynak dosyaya dokunulmaz (spec 0004: masaüstündeki kayıtlar AC01'in
    referans kopyası olarak korunmalıdır). Depoda belge zaten varsa üzerine
    yazılmaz. İçeri alınan anahtarların listesini döndürür.
    """
    imported = []
    for key, filename in LEGACY_FILES.items():
        if read_doc(key) is not None:
            continue
        for folder in legacy_search_dirs():
            source = folder / filename
            if not source.is_file():
                continue
            try:
                with open(source, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception as exc:
                # Okunamayan eski dosya sessizce atlanır; kayıt silinmez.
                logger.warning(f"Legacy import skipped ({source}): {exc}")
                continue
            backup = data_dir() / f"legacy-{filename}"
            try:
                shutil.copy2(source, backup)
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Legacy backup copy failed ({source}): {exc}")
            write_doc(key, payload)
            imported.append(key)
            logger.info(f"Legacy records imported into store: {key} <- {source}")
            break
    return imported
