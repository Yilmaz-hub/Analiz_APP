# Önceden kayıt — altcoin ve altın yöntemleri (2026-10-11, sonuçlara bakılmadan yazıldı)
Motor: uygulamanınki (ertesi açılış, 2,5 ATR stop, zararlı çıkışta 2 gün bekleme, %0,1 komisyon), varlık başına tüm sermaye.
Dönemler: GELİŞTİRME = ilk işlem günü → 2022-06-30; DOĞRULAMA = 2022-07-01 → bugün. Göstergeler tam veriden.
Altcoinler (BTC/ETH dışı 19 kripto):
  M0 C: V1 AL + kapanış > SMA200; çıkış V1 SAT (referans)
  M1 Donchian 50/20: kapanış > önceki 50 günün en yükseği → AL; kapanış < önceki 20 günün en düşüğü → SAT
  M2 Göreli güç: kapanış > SMA200 VE (coin/BTC) > kendi SMA100 → AL; ikisinden biri bozulunca SAT
  M3 C + göreli güç: M0 AL'i yalnız (coin/BTC) > SMA100 iken; çıkış V1 SAT veya oran < SMA100
Altın:
  G0 C, G1 al-tut, G2 RSI(2) dip: kapanış > SMA200 ve RSI2 < 10 → AL; kapanış > SMA5 → SAT; G3 Donchian 50/20
Kabul ölçütü (altcoin): yöntem M0'ı İKİ dönemde de medyan yıllık getiride geçmeli ve medyan en büyük
düşüşü 5 puandan fazla kötüleştirmemeli. Altın: G1 al-tut'u iki dönemde getiri/düşüş oranında geçmeli.
Ölçütü geçen yoksa mevcut yöntem kalır.

# Önceden kayıt — hisseler (2026-10-11, veriye bakılmadan)
Varlıklar: Nasdaq: QQQ + AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO COST NFLX AMD ADBE PEP CSCO INTC;
BIST: XU100 + THYAO ASELS GARAN AKBNK YKBNK ISCTR KCHOL SAHOL BIMAS EREGL TUPRS FROTO SISE TCELL PGSUS.
(Uyarı: bugünkü listelerden seçildi → hayatta kalma yanlılığı; endeksler bu yanlılığı taşımaz.)
Para birimi: BIST fiyatları günlük USDTRY ile dolara çevrilir; kararlar ve sonuçlar dolar serisi üzerinde.
Yöntemler: S0 C (V1 AL + SMA200), S1 Donchian 50/20, S2 RSI(2) dip (SMA200 üstü + RSI2<10, çıkış > SMA5), S3 al-tut.
Dönemler: geliştirme ilk işlem günü → 2022-06-30, doğrulama 2022-07-01 → bugün.
Karar ölçütü (borsa başına ayrı): bir zamanlama yöntemi, medyan (yıllık getiri / |en büyük düşüş|) oranında
al-tut'u İKİ dönemde de geçerse ve medyan yıllık getirisi iki dönemde de ≥ %10 ise seçilir; birden çok
geçen varsa iki dönemin ortalama oranı yüksek olan. Geçen yoksa o borsa için al-tut (zamanlama yok) önerilir.
