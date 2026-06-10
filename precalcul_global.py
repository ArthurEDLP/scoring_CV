"""
Précalcul de l'indicateur global (CV complet ↔ AO complète, Qwen3-8B).

Fait tourner le 8B UNE fois par AO et écrit un JSON {cv_id: cosine_brut} dans
./cache_global. Ensuite THE_HONORED_ONE lit ce JSON au lieu d'embedder en live.

À relancer quand : tu changes d'AO, ou tu ajoutes/modifies des CV.

Usage :
    python precalcul_global.py                       # toutes les AO de ./AO_JSON
    python precalcul_global.py --only CANAL+         # une seule AO (par id)
    python precalcul_global.py --instruction "Étant donné un poste IT, retrouver les profils pertinents"
"""

import argparse
import json
import sys

from CV_AO_Loader import charger_cvs, charger_offres
from score_global import (
    texte_cv_complet, texte_ao_complete,
    embed_ollama, cosinus_brut, chemin_cache_global,
)

MODEL = "qwen3-embedding:8b"


def precalculer_ao(offre, cvs, instruction=None, dossier="./cache_global"):
    """Embedde l'AO + chaque CV, écrit le JSON, retourne (chemin, cosines)."""
    txt_ao = texte_ao_complete(offre["data"])
    if not txt_ao.strip():
        raise RuntimeError(
            f"AO {offre['id']} : texte vide. Vérifie les clés "
            f"Profil/Description/Contexte dans le JSON de l'AO."
        )
    if instruction:                      # asymétrie Qwen3 : instruction côté AO seulement
        txt_ao = f"Instruct: {instruction}\nQuery: {txt_ao}"
    emb_ao = embed_ollama(MODEL, txt_ao)

    cosines = {}
    for cv in cvs:
        emb_cv = embed_ollama(MODEL, texte_cv_complet(cv["data"]))  # CV brut
        cosines[cv["id"]] = round(cosinus_brut(emb_cv, emb_ao), 6)

    chemin = chemin_cache_global(offre["id"], dossier)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps({
            "ao_id":       offre["id"],
            "modele":      MODEL,
            "instruction": instruction,
            "cosines":     cosines,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chemin, cosines


def main():
    p = argparse.ArgumentParser(description="Précalcul de l'indicateur global Qwen3-8B.")
    p.add_argument("--cv",  default="./CV_JSON", help="Dossier des CV (défaut: ./CV_JSON)")
    p.add_argument("--ao",  default="./AO_JSON", help="Dossier des AO (défaut: ./AO_JSON)")
    p.add_argument("--out", default="./cache_global", help="Dossier de sortie (défaut: ./cache_global)")
    p.add_argument("--only", default=None, help="Ne traiter qu'une AO (par id)")
    p.add_argument("--instruction", default=None,
                   help="Instruction Qwen3 (préfixe l'AO seulement). Défaut: aucune (8B brut).")
    args = p.parse_args()

    cvs    = charger_cvs(args.cv)
    offres = charger_offres(args.ao)
    if args.only:
        offres = [o for o in offres if o["id"] == args.only]
    if not cvs or not offres:
        print("❌ Aucun CV ou aucune AO à traiter.")
        sys.exit(1)

    print(f"{len(cvs)} CV · {len(offres)} AO · modèle {MODEL}\n")
    for offre in offres:
        print(f"▸ AO {offre['id']} ...", flush=True)
        try:
            chemin, cosines = precalculer_ao(offre, cvs, args.instruction, args.out)
        except Exception as e:
            print(f"  ✗ {e}\n")
            continue
        classement = sorted(cosines.items(), key=lambda kv: -kv[1])
        for i, (cv_id, c) in enumerate(classement, 1):
            print(f"    {i:>2}. {cv_id:<14} cos={c:.4f}")
        print(f"  💾 {chemin}\n")


if __name__ == "__main__":
    main()
