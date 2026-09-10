"""Varlık listesi iş kuralları (spec 0004).

Liste artık dosyadan değil `storage.py`'deki belge deposundan okunur ve
kayıtları değiştiren her işlem buradaki kurallardan geçer:

* Varlık yalnız kullanıcının açık işlemiyle kaldırılır (R6).
* Aktif pozisyonu, bekleyen emri ya da sorunlu kaydı olan varlık silinemez
  (R6.1, R4.1) — engel `positions.py`'deki tek sınıflandırıcıdan gelir.
* Listedeki son varlık silinemez (G10).
* Toplu sıfırlama ayrıca onay ister ve engelli kayıt varken çalışmaz (R6.2).
* Kayıtlı bir adla ekleme mevcut kaydı sessizce ezemez (R9.1).

Kayıt bulunamadığında **varsayılan liste yazılmaz**: gerçekten boş olan liste
ile erişilemeyen liste birbirinden ayrı durumlardır (R8) ve boş durumda
kendiliğinden kayıt oluşmaz (AC07). Yerleşik liste yalnız kullanıcının açık
isteğiyle yüklenir.
"""
from __future__ import annotations

import positions as positions_module
import storage
from config import DEFAULT_COIN_MAP


def load_assets() -> dict:
    """Kayıtlı varlık listesi; hiç kayıt yoksa boş sözlük.

    Erişim hatasında `storage.StorageAccessError` yükselir; çağıran taraf
    varsayılan listeye düşmez (R8.1).
    """
    doc = storage.read_doc(storage.ASSETS_KEY)
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise storage.StorageAccessError("Varlık kayıtları okunamadı.")
    return doc


def save_assets(assets: dict) -> None:
    """Varlık listesini depoya yazar."""
    storage.write_doc(storage.ASSETS_KEY, assets)


def default_assets() -> dict:
    """Uygulamayla gelen yerleşik liste (yalnız açık istek üzerine kullanılır)."""
    return DEFAULT_COIN_MAP.copy()


def add_asset(assets: dict, name, symbol):
    """Yeni varlık ekler. Döner: (başarılı_mı, mesaj, yeni_liste).

    Başarısız durumda mevcut liste **değiştirilmeden** geri verilir (R9/AC15).
    """
    clean_name = (name or "").strip()
    clean_symbol = (symbol or "").strip()
    if not clean_name or not clean_symbol:
        return False, "İsim ve Sembol boş olamaz!", assets

    for existing in assets:
        if existing.strip().lower() == clean_name.lower():
            return (
                False,
                f"'{existing}' zaten kayıtlı. Mevcut kayıt değiştirilmedi.",
                assets,
            )

    updated = dict(assets)
    updated[clean_name] = clean_symbol
    return True, f"{clean_name} eklendi!", updated


def delete_asset(assets: dict, name, portfolio_positions):
    """Varlığı siler. Döner: (başarılı_mı, mesaj, yeni_liste)."""
    if name not in assets:
        return False, f"'{name}' listede bulunamadı.", assets

    if len(assets) <= 1:
        return (
            False,
            "Listede en az 1 varlık kalmalı! Son varlık silinemez.",
            assets,
        )

    reason = positions_module.deletion_block_reason(name, portfolio_positions)
    if reason:
        return False, reason, assets

    updated = dict(assets)
    del updated[name]
    return True, f"{name} silindi.", updated


def reset_assets(assets: dict, portfolio_positions, confirmed: bool):
    """Listeyi yerleşik varsayılana döndürür (R6.2 / AC11c).

    Kullanıcının ayrıca onayı olmadan ve engelleyici kayıt varken çalışmaz.
    """
    reason = positions_module.reset_block_reason(portfolio_positions)
    if reason:
        return False, reason, assets

    if not confirmed:
        return (
            False,
            "Sıfırlama için önce onay kutusunu işaretleyin.",
            assets,
        )

    return True, "Liste varsayılana döndürüldü.", default_assets()
