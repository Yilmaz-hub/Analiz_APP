# Strateji deneyleri (spec 0019–0021 kanıtı)

Spec 0019 (C), 0020 (kanal 50/20) ve 0021 (hisse al-tut, altın) kararlarının dayandığı deney kodları.
Uygulama kodunu (portfolio_backtest, strategy_candidates, technical_analysis) doğrudan kullanır; sonuçlar
spec'lerin **Kanıt** bölümlerinde özetlidir. Yöntemler ve kabul ölçütü sonuç görülmeden
[PLAN_yontemler.md](PLAN_yontemler.md)'ye yazıldı (ön kayıt).

## Çalıştırma
Betikler `exp/` adlı klasörde ve onun **bir üst** dizininden çalıştırılmak üzere yazıldı (yollar `exp/data_long/...`).

```bash
mkdir -p /tmp/deney && cp -r docs/evidence/strateji-deneyleri /tmp/deney/exp && cd /tmp/deney
python exp/fetch_long.py      # Binance günlük 2018+ (8 kripto) → exp/data_long/
python exp/fetch_more.py      # ek 13 kripto (HBAR, RENDER, …)
python exp/fetch_gold.py      # GC=F (Yahoo chart API) → exp/data_long/XAU_GOLD.pkl
python exp/fetch_stocks.py    # Nasdaq + BIST (BIST USD/TRY ile dolara) → exp/data_stk/
for s in BTC ETH SOL XRP ADA AVAX LINK DOGE; do python exp/dump_dec.py $s-USD; done   # V1 kararları (yavaş)
python exp/per_asset.py       # varlık başına C / V1 / al-tut
python exp/methods.py         # altcoin M0–M3 ve altın G0–G3 (geliştirme ≤2022-06 / doğrulama 2022-07+)
python exp/pf_variants.py     # portföy varyantları; pf_donchian.py, pf_tut.py, stocks.py
```

Veri dosyaları (pkl/json) depoya konmadı; betiklerle yeniden üretilir. Ağ erişimi gerekir
(data-api.binance.vision, query2.finance.yahoo.com).

## Sınırlar
Geçmiş veri; bugün var olan varlıklar (hayatta kalma yanlılığı); kayma yok; hisse al-tut'ta temettü ve
nakit faizi yok. Sonuçlar gelecek getirisi değildir.
