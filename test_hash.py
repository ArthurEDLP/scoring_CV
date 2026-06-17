import traceback
from CV_AO_Loader import charger_cvs
from THE_HONORED_ONE import CACHE_CV   # l'instance déjà configurée

mfu = next(cv for cv in charger_cvs("./CV_JSON") if cv["id"] == "MFU")

# Montre quels champs sont des listes (les coupables potentiels)
for i, exp in enumerate(mfu["data"].get("experiences", [])):
    for champ in ("poste", "entreprise", "date", "details"):
        v = exp.get(champ)
        if not isinstance(v, (str, type(None))):
            print(f"exp#{i}.{champ} = {type(v).__name__} : {str(v)[:80]}")
for j, t in enumerate(mfu["data"].get("competences_techniques", [])):
    if not isinstance(t, str):
        print(f"techno#{j} = {type(t).__name__} : {str(t)[:80]}")

print("\n--- appel obtenir(MFU) ---")
try:
    CACHE_CV.obtenir(mfu)
    print("✓ OK, cache écrit")
except Exception:
    traceback.print_exc()