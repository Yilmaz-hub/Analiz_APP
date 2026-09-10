"""Pozisyon kayıtlarının sınıflandırılması (spec 0004, R4.1).

Bir pozisyonun "aktif" sayılıp sayılmadığı iki yerde birden kullanılır:
görünürlükte (hangi tabloda listeleneceği) ve silme engelinde (varlığın
silinip silinemeyeceği). Spec bu iki yorumun **aynı** olmasını şart koşar, bu
yüzden tek sınıflandırıcı vardır ve her iki taraf da buradan beslenir.

Sınıflar:

* ``AKTIF``    — durumu ACTIVE ve kalan miktarı sıfırdan büyük (G01).
* ``BEKLEYEN`` — henüz başlamamış, tutarı kilitli emir.
* ``KAPALI``   — kalan miktarı sıfıra inmiş pozisyon; geçmiş kaydı korunur.
* ``SORUNLU``  — miktarı okunamayan, eksik ya da durumu tanınmayan kayıt.

**SORUNLU kayıt kapatılmış sayılmaz** (R4.1): korunur, boş kayıtla
değiştirilmez, kullanıcıya sorunlu olarak gösterilir ve ilişkili varlığın
silinmesini engeller. Belirsizliği "kapalı" yönünde yorumlamak, kaybın
kapısını açık bırakırdı.
"""
from __future__ import annotations

AKTIF = "aktif"
BEKLEYEN = "bekleyen"
KAPALI = "kapali"
SORUNLU = "sorunlu"

#: Varlığın silinmesini engelleyen sınıflar — öncelik sırasıyla.
ENGELLEYEN = (AKTIF, BEKLEYEN, SORUNLU)

_CLOSED_STATUSES = ("CLOSED_TP", "CLOSED_SL", "CLOSED")
_QUANTITY_KEY = "Adet"

_REASONS = {
    AKTIF: (
        "'{ad}' silinemez: bu varlığın aktif pozisyonu var. "
        "Önce pozisyonu kapatın."
    ),
    BEKLEYEN: (
        "'{ad}' silinemez: bu varlığın bekleyen emri var. "
        "Kilitli tutar korunuyor; önce emri iptal edin."
    ),
    SORUNLU: (
        "'{ad}' silinemez: bu varlığa ait, bilgileri okunamayan bir kayıt var. "
        "Kayıt korunuyor; silmeden önce incelenmesi gerekir."
    ),
}


def read_quantity(pos):
    """Kalan miktarı sayı olarak döndürür; okunamıyorsa None."""
    if not isinstance(pos, dict) or _QUANTITY_KEY not in pos:
        return None
    raw = pos.get(_QUANTITY_KEY)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def asset_name(pos):
    """Kaydın işaret ettiği varlık adı; okunamıyorsa None."""
    if not isinstance(pos, dict):
        return None
    name = pos.get("Coin")
    if isinstance(name, str) and name.strip():
        return name
    return None


def classify_position(pos) -> str:
    """Kaydı AKTIF / BEKLEYEN / KAPALI / SORUNLU sınıflarından birine koyar."""
    if not isinstance(pos, dict) or asset_name(pos) is None:
        return SORUNLU

    # Durum alanı yoksa eski kayıtlarda olduğu gibi ACTIVE varsayılır; bu,
    # uygulamanın bugünkü davranışıdır ve kaydı sorunlu saymak için gerekçe
    # değildir.
    status = pos.get("Status", "ACTIVE")
    if not isinstance(status, str):
        return SORUNLU
    status = status.strip().upper()

    if status == "PENDING":
        return BEKLEYEN
    if status in _CLOSED_STATUSES:
        return KAPALI
    if status != "ACTIVE":
        return SORUNLU

    quantity = read_quantity(pos)
    if quantity is None or quantity < 0:
        return SORUNLU
    return KAPALI if quantity == 0 else AKTIF


def group_positions(positions) -> dict[str, list]:
    """Kayıtları sınıflarına göre ayırır (görünürlük ve engel aynı kaynaktan)."""
    groups: dict[str, list] = {AKTIF: [], BEKLEYEN: [], KAPALI: [], SORUNLU: []}
    if not isinstance(positions, list):
        return groups
    for pos in positions:
        groups[classify_position(pos)].append(pos)
    return groups


def blocking_kind_for_asset(asset, positions):
    """Varlığın silinmesini engelleyen sınıf; engel yoksa None.

    Adı okunamayan SORUNLU kayıtlar hiçbir varlığa bağlanamaz; bunlar tek tek
    varlık silmeyi engellemez (toplu sıfırlamayı engeller, bkz.
    `blocking_kinds`).
    """
    if not isinstance(positions, list):
        return None
    for kind in ENGELLEYEN:
        for pos in positions:
            if asset_name(pos) == asset and classify_position(pos) == kind:
                return kind
    return None


def deletion_block_reason(asset, positions):
    """Silme engelinin kullanıcıya gösterilecek gerekçesi; engel yoksa None."""
    kind = blocking_kind_for_asset(asset, positions)
    if kind is None:
        return None
    return _REASONS[kind].format(ad=asset)


def blocking_kinds(positions) -> list[str]:
    """Portföydeki tüm engelleyici sınıflar (toplu sıfırlama için)."""
    groups = group_positions(positions)
    return [kind for kind in ENGELLEYEN if groups[kind]]


def reset_block_reason(positions):
    """Toplu sıfırlamayı engelleyen gerekçe; engel yoksa None."""
    kinds = blocking_kinds(positions)
    if not kinds:
        return None
    labels = {
        AKTIF: "aktif pozisyon",
        BEKLEYEN: "bekleyen emir",
        SORUNLU: "bilgileri okunamayan kayıt",
    }
    listed = ", ".join(labels[kind] for kind in kinds)
    return (
        f"Sıfırlama yapılamaz: portföyde {listed} bulunuyor. "
        "Bu kayıtlar korunuyor."
    )


def build_active_rows(active_positions, price_lookup):
    """Aktif pozisyon tablosunun satırlarını ve toplam değerini üretir.

    `price_lookup(coin)` canlı fiyatı verir; fiyat alınamazsa (0) giriş fiyatı
    kullanılır. Yatırım tutarı sıfır ya da okunamaz olduğunda yüzde hesabı
    yapılmaz — bir bölme hatası tüm kayıtların görünmesini engellerdi.

    Kayıtların okunmasından listenin görünmesine kadar geçen yol AC17'de
    ölçülür; bu işlev o ölçümün arayüzden bağımsız durağıdır.
    """
    rows, total = [], 0.0
    for item in active_positions:
        quantity = read_quantity(item) or 0.0
        entry = item.get("Giriş", 0.0)
        price = price_lookup(item.get("Coin"))
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 0.0
        if price == 0:
            try:
                price = float(entry)
            except (TypeError, ValueError):
                price = 0.0
        value = quantity * price
        total += value
        try:
            invested = float(item.get("Yatırım", 0.0))
        except (TypeError, ValueError):
            invested = 0.0
        profit = value - invested
        pct = f"%{(profit / invested) * 100:.2f}" if invested else "-"
        rows.append({
            "Coin": item.get("Coin"), "Giriş": entry, "Adet": item.get("Adet"),
            "Değer ($)": value, "Kar/Zarar ($)": profit, "Kar/Zarar (%)": pct,
        })
    return rows, total
