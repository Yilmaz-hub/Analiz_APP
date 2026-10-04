# Spec 0005 — Adım 5–7 ekran kanıtı (QA bulgusu B19)

Ekran görüntüleri `harness.py` ile üretildi: gerçek arayüz modüllerinin gösterim satırlarını
(`performance_ui`, `candidate_ui`, `risk_ui`, `protection_ui`, `forward_ui`) sabit veriyle çizer;
kayıt deposu geçici klasöre yönlendirilir, gerçek kayıtlara dokunulmaz. Gerçek `app.py` akışı ayrıca
`AppTest` smoke testleriyle sürülür (`tests/test_performance_ui.py`, `tests/test_qa_*.py`).

| Dosya | Kanıtlanan |
|---|---|
| `regime.png` | R04, AC89 — piyasa koşulu filtresi: sürüm etiketi (REJIM-1), işlem sayısı ve beklenti, net getiri ve düşüşün "hesaplanamıyor" notu |
| `candidates.png` | AC01, AC16, AC17, AC48, AC85, AC96, AC98 — aktif strateji (kullanıcı seçimi), piyasa koşulu + sürümü, aday geçmişi, bilinmeyen maliyette "yetersiz veri", "Aktif yap" / "V1'e dön" |
| `risk.png` | AC23, AC30, AC72, AC93, AC94 — dönem başlangıcı, sıfırlama ve sınır değişikliği kayıtları, gerçek nakitle miktar, bilinmeyen maliyet uyarısı, açık pozisyonun stop ve son uyarısı |
| `protection.png` | R13, AC95 — V1 referansı, sabit hedef, iz süren stop, kademeli çıkış yan yana; sanal olduğu notu |
| `forward.png` | AC35, AC36, AC38, AC71, AC99, AC103 — son (başarısız) çalışma uyarısı, yeterlilik ilerlemesi, eksik görünen günler, zamanında / sonradan oluşturuldu, revizyon, varsayımlar |

Yeniden üretim: `python docs/evidence/0005/adim5-7/capture.py` (Playwright ve önceden kurulu Chromium
gerekir) ya da `streamlit run docs/evidence/0005/adim5-7/harness.py` ve
`?state=regime|candidates|risk|protection|forward`.
