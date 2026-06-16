"""
Pré-traitement des CVs bruts vers le format propre attendu par le pipeline.

Version déterministe, SANS LLM : on ne touche qu'aux compétences techniques
(éclatement des préfixes catégoriels et des séparateurs, déduplication).
Tout le reste du CV — expériences, savoir-faire/être, formations, langues —
est recopié TEL QUEL depuis le brut.

Usage CLI :
    python pretraiter_cv.py CV_JSON_brutes/JL.json
    python pretraiter_cv.py CV_JSON_brutes/JL.json --output CV_JSON/
    python pretraiter_cv.py CV_JSON_brutes/*.json --output CV_JSON/ --force
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────
# Nettoyage des technos (déterministe, fiable)
# ─────────────────────────────────────────────────────────────────────

# Préfixe catégoriel : tout ce qui précède le premier ":" si < 40 caractères
# (évite de matcher des phrases entières contenant un ":")
_RE_PREFIXE = re.compile(r"^[^:]{1,40}:\s*")


def _split_hors_parentheses(texte: str) -> List[str]:
    """
    Split sur |, ;, /, et virgule, mais SEULEMENT en dehors des parenthèses.
    Ainsi 'Agile (Scrum, Kanban) | Lean' donne ['Agile (Scrum, Kanban)', 'Lean'].
    """
    morceaux = []
    courant = []
    niveau = 0   # profondeur de parenthèses

    for c in texte:
        if c == "(":
            niveau += 1
            courant.append(c)
        elif c == ")":
            niveau = max(0, niveau - 1)
            courant.append(c)
        elif niveau == 0 and c in "|;/,":
            if courant:
                morceaux.append("".join(courant))
                courant = []
        else:
            courant.append(c)

    if courant:
        morceaux.append("".join(courant))

    return morceaux


def _eclater_technos(technos_brutes: List[str]) -> List[str]:
    """
    Prend une liste de strings 'sales' (avec préfixes et séparateurs) et
    retourne une liste propre de technos individuelles.

    Exemples :
        'Méthodologies : Agile (Scrum, Kanban) | Lean Six Sigma'
            → ['Agile (Scrum, Kanban)', 'Lean Six Sigma']
        'Python | SQL'                      → ['Python', 'SQL']
        'Cloud : AWS'                       → ['AWS']

    Dédup insensible à la casse, conserve l'orthographe de la 1ère occurrence.
    """
    resultat = []
    vus = set()

    for entree in technos_brutes:
        if not isinstance(entree, str):
            continue
        # 1. Retirer le préfixe catégoriel "X : "
        nettoye = _RE_PREFIXE.sub("", entree, count=1)
        # 2. Éclater sur les séparateurs hors parenthèses
        morceaux = _split_hors_parentheses(nettoye)
        # 3. Trim, filtrer vides, dédupliquer
        for m in morceaux:
            m = m.strip()
            # Filtre minimum : au moins 2 caractères, max 60 (sinon c'est du blabla)
            if 2 <= len(m) <= 60 and m.lower() not in vus:
                resultat.append(m)
                vus.add(m.lower())

    return resultat


# ─────────────────────────────────────────────────────────────────────
# Structuration : on ne touche QUE les technos, le reste est recopié
# ─────────────────────────────────────────────────────────────────────

def structurer_cv(cv_brut: Dict) -> Dict:
    """Recopie le CV brut à l'identique et remplace competences_techniques
    par sa version éclatée/dédupliquée. Aucun autre champ n'est modifié."""
    cv_propre = dict(cv_brut)
    cv_propre["competences_techniques"] = _eclater_technos(
        cv_brut.get("competences_techniques", []) or []
    )
    return cv_propre


# ─────────────────────────────────────────────────────────────────────
# Wrapper CLI
# ─────────────────────────────────────────────────────────────────────

def _traiter_fichier(chemin_in: Path, dossier_out: Path) -> bool:
    try:
        with open(chemin_in, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ {chemin_in.name} : impossible de lire le JSON ({e})")
        return False

    if isinstance(data, list):
        data = data[0]
    cv_brut = data.get("data", data)
    enveloppe = data if "data" in data else None

    print(f"  -> {chemin_in.name} ... ", end="", flush=True)
    cv_propre = structurer_cv(cv_brut)

    if enveloppe is not None:
        sortie = dict(enveloppe)
        sortie["data"] = cv_propre
    else:
        sortie = cv_propre

    chemin_out = dossier_out / chemin_in.name
    with open(chemin_out, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)

    nb_technos = len(cv_propre.get("competences_techniques", []))
    nb_exp     = len(cv_propre.get("experiences", []))
    print(f"✓ {nb_technos} technos, {nb_exp} expériences")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Structure des CVs bruts (nettoyage déterministe des technos, sans LLM)."
    )
    parser.add_argument("fichiers", nargs="+", help="Fichier(s) CV à traiter")
    parser.add_argument("--output", "-o", default="./CV_JSON")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    dossier_out = Path(args.output)
    dossier_out.mkdir(parents=True, exist_ok=True)

    print(f"📂 Sortie : {dossier_out.resolve()}\n")

    nb_ok = nb_skip = nb_ko = 0

    for arg in args.fichiers:
        chemin = Path(arg)
        if not chemin.exists():
            print(f"  ✗ {arg} : fichier introuvable")
            nb_ko += 1
            continue

        chemin_out = dossier_out / chemin.name
        if chemin_out.exists() and not args.force:
            print(f"  ⏭️  {chemin.name} : déjà traité (--force pour écraser)")
            nb_skip += 1
            continue

        if _traiter_fichier(chemin, dossier_out):
            nb_ok += 1
        else:
            nb_ko += 1

    print(f"\n📊 Bilan : {nb_ok} OK, {nb_skip} ignorés, {nb_ko} erreurs")
    sys.exit(0 if nb_ko == 0 else 1)


if __name__ == "__main__":
    main()