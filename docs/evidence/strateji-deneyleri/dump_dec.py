import json, os, pickle, sys
sys.path.insert(0, "/home/user/Analiz_APP")
HERE = os.path.dirname(os.path.abspath(__file__))
from technical_analysis import build_v1_decisions
from weight_profiles import get_weights_for_symbol
sym = sys.argv[1]
df = pickle.load(open(f"{HERE}/data_long/{sym}.pkl", "rb"))
dec = build_v1_decisions(df, weights=get_weights_for_symbol(sym), include_ml=False)
json.dump({str(k.date()): v for k, v in dec.items()}, open(f"{HERE}/data_long/{sym}.dec.json", "w"))
print(sym, len(dec))
