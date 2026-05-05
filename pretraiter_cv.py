"""
Pré-traitement des CVs bruts vers le format propre attendu par le pipeline.

Utilise Ollama en local avec qwen2.5:7b-instruct.

Format d'entrée (CV brut, tel que sorti d'un extracteur PDF) :
    {
      "id": "JL",
      "competences_techniques": ["Langages & Scripting : Python |SQL |", ...],
      "savoir_faire/savoir_etre": [...],
      "experiences": [{"poste": "...", "date": "...", ...}],
      "formations": [...],
      "langues": [...]
    }

Format de sortie (CV nettoyé) :
    {
      "id": "JL",
      "competences_techniques": ["Python", "SQL", ...],   # éclatées + dédup
      "savoir_faire/savoir_etre": [...],                  # conservé
      "experiences": [
        {
          "poste": "Data Scientist",
          "entreprise": "TOTAL",
          "date": "01/2023 - 12/2025",       # MM/YYYY normalisé
          "details": "..."
        }
      ],
      "formations": [...],                                # conservé
      "langues": [...]                                    # conservé
    }

Usage CLI :
    python pretraiter_cv.py CV_JSON_brutes/JL.json
    python pretraiter_cv.py CV_JSON_brutes/JL.json --output CV_JSON/
    python pretraiter_cv.py CV_JSON_brutes/JL.json --force

Pré-requis :
    - Ollama lancé (auto sur Windows)
    - qwen2.5:7b-instruct téléchargé
    - pip install requests
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, List

import requests


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5:7b-instruct"
TIMEOUT_SEC   = 180        # un CV peut être plus long qu'une AO
MAX_RETRIES   = 2


# ─────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans le nettoyage et la structuration de \
CVs. Le CV en entrée a été extrait d'un PDF et peut contenir des artefacts d'extraction \
(séparateurs |, retours à la ligne dans des champs uniques, préfixes catégoriels collés \
aux technos, etc.). Tu réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après, \
sans bloc markdown.

Tu dois produire un JSON avec ces champs :
  - "competences_techniques" : liste de strings, UNE techno par entrée, sans préfixe
  - "savoir_faire/savoir_etre" : liste de strings (conserver le contenu existant, \
nettoyer les artefacts d'extraction)
  - "experiences" : liste d'objets avec {poste, entreprise, date, details}
  - "formations" : liste (conserver telle quelle si propre)
  - "langues" : liste (conserver telle quelle si propre)

RÈGLES STRICTES :

1. TECHNOS — la règle la plus importante :
   - Si une entrée contient PLUSIEURS technos séparées par |, ;, , ou /, ÉCLATER en \
plusieurs entrées séparées.
   - Si une entrée a un préfixe catégoriel ("Langages & Scripting :", "Cloud :", "BDD :", \
"Outils :", etc.), RETIRER le préfixe et ne garder QUE la techno.
   - GARDER L'ORTHOGRAPHE EXACTE de la techno telle qu'écrite dans le CV (ne pas changer \
la casse, ne pas reformater).
   - Dédupliquer : si la même techno apparaît plusieurs fois, ne la mettre qu'une seule fois.
   - N'ajouter AUCUNE techno qui n'est pas dans le CV.
   - Une techno doit être un nom propre (Python, SQL, AWS, Dataiku) — pas un soft skill.

2. EXPÉRIENCES :
   - Format de date attendu : "MM/YYYY - MM/YYYY" ou "MM/YYYY - Aujourd'hui"
   - Si une seule date est présente (ex: "01/2015"), la garder telle quelle
   - Si la date est en format "Mars 2019", convertir en "03/2019"
   - "Aujourd'hui", "présent", "now" → "Aujourd'hui"
   - Conserver les champs poste, entreprise, details existants

3. NE PAS INVENTER :
   - Si un champ est absent du CV, le mettre à [] (liste vide) ou "" (string vide)
   - Ne pas générer de contenu plausible mais absent du CV
"""

EXEMPLE_FEW_SHOT = """EXEMPLE COMPLET :

Input :
{
  "id": "EXEMPLE_CANDIDAT",
  "competences_techniques": [
    "Langages & Scripting : Python |SQL | Java",
    "Cloud : AWS |GCP",
    "BDD : PostgreSQL"
  ],
  "savoir_faire/savoir_etre": [
    "Esprit d'équipe",
    "Autonomie"
  ],
  "experiences": [
    {
      "poste": "Data Engineer",
      "date": "Mars 2020 - Aujourd hui",
      "entreprise": "TotalEnergies",
      "details": "Pipeline ETL"
    }
  ],
  "formations": [],
  "langues": []
}

Output :
{"competences_techniques":["Python","SQL","Java","AWS","GCP","PostgreSQL"],"savoir_faire/savoir_etre":["Esprit d'équipe","Autonomie"],"experiences":[{"poste":"Data Engineer","entreprise":"TotalEnergies","date":"03/2020 - Aujourd'hui","details":"Pipeline ETL"}],"formations":[],"langues":[]}
"""


def _construire_prompt(cv_brut: Dict) -> str:
    # On ne passe au LLM que les champs pertinents (id ignoré, on le reprend après)
    payload = {
        k: v for k, v in cv_brut.items()
        if k in (
            "competences_techniques",
            "savoir_faire/savoir_etre",
            "experiences",
            "formations",
            "langues",
        )
    }
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{EXEMPLE_FEW_SHOT}\n\n"
        f"À TOI MAINTENANT.\n\n"
        f"Input :\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Output :\n"
    )


# ─────────────────────────────────────────────────────────────────────
# Appel Ollama
# ─────────────────────────────────────────────────────────────────────

def _appeler_ollama(prompt: str) -> str:
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.1},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Impossible de joindre Ollama sur localhost:11434.\n"
            "Vérifie qu'Ollama tourne (icône dans la barre des tâches Windows)."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Erreur HTTP Ollama : {e}")


def _parser_reponse(reponse_brute: str) -> Dict:
    txt = reponse_brute.strip()

    # Retire fences markdown si Qwen en a mis
    if txt.startswith("```"):
        lignes = txt.split("\n")
        if lignes[0].startswith("```"):
            lignes = lignes[1:]
        if lignes and lignes[-1].strip().startswith("```"):
            lignes = lignes[:-1]
        txt = "\n".join(lignes)

    return json.loads(txt)


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

def _valider_extraction(extrait: Dict) -> Optional[str]:
    """
    Vérifie que les champs sont présents et bien typés.
    Retourne None si OK, sinon un message d'erreur.
    """
    champs_listes = [
        "competences_techniques",
        "savoir_faire/savoir_etre",
        "experiences",
        "formations",
        "langues",
    ]
    for champ in champs_listes:
        if champ not in extrait:
            return f"champ manquant : '{champ}'"
        if not isinstance(extrait[champ], list):
            return f"champ '{champ}' n'est pas une liste"

    # Toutes les technos doivent être des strings non vides et SANS séparateurs
    # (vérification anti-régression : si Qwen oublie d'éclater, on retente)
    for i, t in enumerate(extrait["competences_techniques"]):
        if not isinstance(t, str) or not t.strip():
            return f"techno #{i} invalide : {t!r}"
        # Si on voit encore des séparateurs, c'est que l'éclatement a foiré
        if any(sep in t for sep in ["|", ";"]):
            return f"techno #{i} contient encore un séparateur : {t!r}"

    # Chaque expérience doit avoir au moins poste et date
    for i, exp in enumerate(extrait["experiences"]):
        if not isinstance(exp, dict):
            return f"experience #{i} n'est pas un objet"
        if "poste" not in exp or "date" not in exp:
            return f"experience #{i} sans poste ou date"
        # Garantir que les sous-champs sont des strings
        for sous_champ in ("poste", "entreprise", "date", "details"):
            if sous_champ in exp and not isinstance(exp[sous_champ], str):
                exp[sous_champ] = str(exp[sous_champ])

    return None


# ─────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────

def structurer_cv(cv_brut: Dict) -> Dict:
    """
    Prend un CV brut et retourne sa version nettoyée.
    L'id est conservé, et tout champ non listé reste tel quel.
    """
    if "id" not in cv_brut:
        raise ValueError("Le CV doit avoir un champ 'id'")

    prompt = _construire_prompt(cv_brut)
    derniere_erreur = None

    for tentative in range(1, MAX_RETRIES + 2):
        try:
            reponse_brute = _appeler_ollama(prompt)
            extrait       = _parser_reponse(reponse_brute)
            erreur        = _valider_extraction(extrait)

            if erreur:
                derniere_erreur = erreur
                continue

            # Fusion : on part du CV brut (pour conserver id et tout champ extra),
            # puis on ÉCRASE avec les champs nettoyés
            cv_propre = dict(cv_brut)
            cv_propre.update({
                "competences_techniques":     extrait["competences_techniques"],
                "savoir_faire/savoir_etre":   extrait["savoir_faire/savoir_etre"],
                "experiences":                extrait["experiences"],
                "formations":                 extrait["formations"],
                "langues":                    extrait["langues"],
            })
            return cv_propre

        except json.JSONDecodeError as e:
            derniere_erreur = f"JSON invalide : {e}"
        except Exception as e:
            derniere_erreur = str(e)

    raise RuntimeError(
        f"Échec après {MAX_RETRIES + 1} tentatives. "
        f"Dernière erreur : {derniere_erreur}"
    )


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

    # Tolère le format [{"data": {...}}] vu dans CV_AO_Loader
    if isinstance(data, list):
        data = data[0]
    cv_brut = data.get("data", data)

    # On préserve la structure d'enveloppe d'origine (data, file, etc.) si présente
    enveloppe = data if "data" in data else None

    print(f"  → {chemin_in.name} ... ", end="", flush=True)
    try:
        cv_propre = structurer_cv(cv_brut)
    except Exception as e:
        print(f"✗ {e}")
        return False

    # Reconstruction du fichier de sortie en respectant le format d'origine
    if enveloppe is not None:
        # Format: {"data": {...}, "file": "..."}
        sortie = dict(enveloppe)
        sortie["data"] = cv_propre
    else:
        sortie = cv_propre

    chemin_out = dossier_out / chemin_in.name
    with open(chemin_out, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)

    nb_technos = len(cv_propre["competences_techniques"])
    nb_exp     = len(cv_propre["experiences"])
    print(f"✓ {nb_technos} technos, {nb_exp} expériences")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Structure des CVs bruts via Ollama (Qwen 2.5)."
    )
    parser.add_argument(
        "fichiers",
        nargs="+",
        help="Fichier(s) CV à traiter (ex: CV_JSON_brutes/JL.json). "
             "Wildcards non supportés en PowerShell : utiliser un foreach.",
    )
    parser.add_argument(
        "--output", "-o",
        default="./CV_JSON",
        help="Dossier de sortie (créé si absent). Défaut : ./CV_JSON",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-traite même si le fichier de sortie existe déjà",
    )
    args = parser.parse_args()

    dossier_out = Path(args.output)
    dossier_out.mkdir(parents=True, exist_ok=True)

    print(f"📂 Sortie : {dossier_out.resolve()}")
    print(f"🤖 Modèle : {OLLAMA_MODEL}\n")

    # Test connexion Ollama
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        print("❌ Ollama injoignable. Vérifie l'icône Ollama dans la barre des tâches.")
        sys.exit(1)

    nb_ok = 0
    nb_skip = 0
    nb_ko = 0

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
