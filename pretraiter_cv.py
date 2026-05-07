"""
Pré-traitement des CVs bruts vers le format propre attendu par le pipeline.

Utilise Ollama en local avec phi3.5 + un SCHÉMA JSON STRICT.

Le schéma force le modèle à produire un JSON conforme au niveau du
décodeur de tokens : pas de champ manquant possible, pas de format cassé.

Usage CLI :
    python pretraiter_cv.py CV_JSON_brutes/JL.json
    python pretraiter_cv.py CV_JSON_brutes/JL.json --output CV_JSON/
    python pretraiter_cv.py CV_JSON_brutes/JL.json --force
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import requests


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "phi3.5"        # Microsoft Phi-3.5 mini, réputé en JSON
TIMEOUT_SEC   = 300
MAX_RETRIES   = 1

TAILLE_MAX_CV = 12000


# ─────────────────────────────────────────────────────────────────────
# Schéma JSON imposé au modèle
# ─────────────────────────────────────────────────────────────────────
# Le paramètre `format` d'Ollama accepte un schéma JSON Schema.
# Le modèle est CONTRAINT au niveau du décodeur de respecter ce schéma.

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "competences_techniques": {
            "type": "array",
            "items": {"type": "string"},
        },
        "savoir_faire": {
            "type": "array",
            "items": {"type": "string"},
        },
        "savoir_etre": {
            "type": "array",
            "items": {"type": "string"},
        },
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "poste":      {"type": "string"},
                    "entreprise": {"type": "string"},
                    "date":       {"type": "string"},
                    "details":    {"type": "string"},
                },
                "required": ["poste", "entreprise", "date", "details"],
            },
        },
        "formations": {
            "type": "array",
            "items": {"type": "object"},
        },
        "langues": {
            "type": "array",
            "items": {"type": "object"},
        },
    },
    "required": [
        "competences_techniques",
        "savoir_faire",
        "savoir_etre",
        "experiences",
        "formations",
        "langues",
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans le nettoyage et la structuration de \
CVs. Le CV en entrée a été extrait d'un PDF et peut contenir des artefacts d'extraction \
(séparateurs |, retours à la ligne, préfixes catégoriels collés aux technos).

Tu dois produire un JSON avec ces champs (TOUS obligatoires, même vides) :
  - competences_techniques : liste de strings, UNE techno par entrée
  - savoir_faire : liste de strings
  - savoir_etre : liste de strings
  - experiences : liste d'objets {poste, entreprise, date, details}
  - formations : liste (vide [] si absentes)
  - langues : liste (vide [] si absentes)

RÈGLES STRICTES :

1. TECHNOS — règle critique :
   - Si une entrée contient PLUSIEURS technos séparées par |, ;, , ou /, ÉCLATER.
   - Retirer les préfixes catégoriels ("Langages :", "Cloud :", "BDD :", etc.).
   - GARDER L'ORTHOGRAPHE du CV (ne pas changer la casse).
   - Dédupliquer.
   - N'ajouter AUCUNE techno absente du CV.
   - Pas de soft skill ou de savoir_faire dans les technos (autonomie, rigueur, Optimisation, etc.).

2. savoir_etre — règle critique :
   - GARDER L'ORTHOGRAPHE du CV (ne pas changer la casse).
   - Dédupliquer.
   - N'ajouter AUCUN savoir être absent du CV.
   - Pas de techno ou de savoir_faire dans les savoir_etre (SQL, Optimisation, etc.).

3. savoir_faire — règle critique :
   - GARDER L'ORTHOGRAPHE du CV (ne pas changer la casse).
   - Dédupliquer.
   - N'ajouter AUCUN savoir faire absent du CV.
   - Pas de techno ou de soft_skill dans les savoir_faire (SQL, autonomie, etc.).

4. EXPÉRIENCES :
   - Format de date : "MM/YYYY - MM/YYYY" ou "MM/YYYY - Aujourd'hui"
   - "Mars 2019" -> "03/2019"
   - "présent", "now", "current" -> "Aujourd'hui"
   - Si un sous-champ est absent du CV, mettre une string vide ""
   - Une même expérience ne peut apparaître deux fois.

5. NE PAS INVENTER :
   - Champ absent -> liste vide [] (jamais null)
   - Pas de contenu plausible mais absent du CV
"""

EXEMPLE_FEW_SHOT = """EXEMPLE :

Input :
{
  "competences_techniques": [
    "Langages & Scripting : Python |SQL | Java",
    "Cloud : AWS"
  ],
  "savoir_faire/savoir_etre": ["IA générative", "Optimisation du pré-processing", "Esprit d'équipe", "Rigueur"],
  "experiences": [
    {
      "poste": "Data Engineer",
      "date": "Mars 2020 - Aujourd hui",
      "entreprise": "TotalEnergies",
      "details": ["Pipeline ETL", " "]
    }
  ],
  "formations": [],
  "langues": []
}

Output :
{"competences_techniques":["Python","SQL","Java","AWS"],"savoir_etre":["Esprit d'équipe", "Rigueur"],"savoir_faire":["IA générative", "Optimisation du pré-processing"],"experiences":[{"poste":"Data Engineer","entreprise":"TotalEnergies","date":"03/2020 - Aujourd'hui","details":["Pipeline ETL"," "]}],"formations":[],"langues":[]}
"""


def _tronquer_si_trop_long(payload: Dict) -> Dict:
    """Coupe les `details` des expériences anciennes si CV trop volumineux."""
    taille = len(json.dumps(payload, ensure_ascii=False))
    if taille <= TAILLE_MAX_CV:
        return payload

    payload_t = json.loads(json.dumps(payload, ensure_ascii=False))
    for exp in reversed(payload_t.get("experiences", [])):
        if isinstance(exp, dict) and exp.get("details"):
            exp["details"] = ""
            taille = len(json.dumps(payload_t, ensure_ascii=False))
            if taille <= TAILLE_MAX_CV:
                break
    return payload_t


def _construire_prompt(cv_brut: Dict) -> str:
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
    payload = _tronquer_si_trop_long(payload)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{EXEMPLE_FEW_SHOT}\n\n"
        f"À TOI MAINTENANT.\n\n"
        f"Input :\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Output :\n"
    )


# ─────────────────────────────────────────────────────────────────────
# Appel Ollama avec schéma JSON
# ─────────────────────────────────────────────────────────────────────

def _appeler_ollama(prompt: str) -> str:
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "format":  JSON_SCHEMA,        # ← schéma strict, pas juste "json"
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
    if txt.startswith("```"):
        lignes = txt.split("\n")
        if lignes[0].startswith("```"):
            lignes = lignes[1:]
        if lignes and lignes[-1].strip().startswith("```"):
            lignes = lignes[:-1]
        txt = "\n".join(lignes)
    return json.loads(txt)


# ─────────────────────────────────────────────────────────────────────
# Validation (le schéma garantit déjà le format, on vérifie le métier)
# ─────────────────────────────────────────────────────────────────────

def _valider_extraction(extrait: Dict) -> Optional[str]:
    for i, t in enumerate(extrait["competences_techniques"]):
        if not t.strip():
            return f"techno #{i} vide"
        if any(sep in t for sep in ["|", ";"]):
            return f"techno #{i} contient encore un séparateur : {t!r}"

    for i, exp in enumerate(extrait["experiences"]):
        if not exp.get("poste", "").strip():
            return f"experience #{i} sans poste"
        if not exp.get("date", "").strip():
            return f"experience #{i} sans date"

    return None


# ─────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────

def structurer_cv(cv_brut: Dict, debug: bool = False) -> Dict:
    if "id" not in cv_brut:
        raise ValueError("Le CV doit avoir un champ 'id'")

    prompt = _construire_prompt(cv_brut)
    derniere_erreur = None
    derniere_reponse = None

    for tentative in range(1, MAX_RETRIES + 2):
        try:
            reponse_brute = _appeler_ollama(prompt)
            derniere_reponse = reponse_brute

            if debug:
                print(f"\n--- Tentative {tentative} : réponse brute du LLM ---")
                print(reponse_brute[:2000])
                print("---\n")

            extrait = _parser_reponse(reponse_brute)
            erreur  = _valider_extraction(extrait)

            if erreur:
                derniere_erreur = erreur
                if debug:
                    print(f"  -> validation échouée : {erreur}\n")
                continue

            cv_propre = dict(cv_brut)
            cv_propre.update({
                "competences_techniques":     extrait["competences_techniques"],
                "savoir_faire":               extrait["savoir_faire"],
                "savoir_etre":                extrait["savoir_etre"],
                "experiences":                extrait["experiences"],
                "formations":                 extrait["formations"],
                "langues":                    extrait["langues"],
            })
            return cv_propre

        except json.JSONDecodeError as e:
            derniere_erreur = f"JSON invalide : {e}"
        except Exception as e:
            derniere_erreur = str(e)

    # Échec final : on remonte aussi la dernière réponse pour faciliter le debug
    msg = f"Échec après {MAX_RETRIES + 1} tentatives. Dernière erreur : {derniere_erreur}"
    if derniere_reponse and not debug:
        msg += f"\n  Dernière réponse du LLM (200 premiers car.) : {derniere_reponse[:200]}..."
    raise RuntimeError(msg)


# ─────────────────────────────────────────────────────────────────────
# Wrapper CLI
# ─────────────────────────────────────────────────────────────────────

def _traiter_fichier(chemin_in: Path, dossier_out: Path, debug: bool = False) -> bool:
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
    try:
        cv_propre = structurer_cv(cv_brut, debug=debug)
    except Exception as e:
        print(f"✗ {e}")
        return False

    if enveloppe is not None:
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
        description="Structure des CVs bruts via Ollama (schéma JSON strict)."
    )
    parser.add_argument("fichiers", nargs="+", help="Fichier(s) CV à traiter")
    parser.add_argument("--output", "-o", default="./CV_JSON")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Affiche la réponse brute du LLM à chaque tentative")
    args = parser.parse_args()

    dossier_out = Path(args.output)
    dossier_out.mkdir(parents=True, exist_ok=True)

    print(f"📂 Sortie : {dossier_out.resolve()}")
    print(f"🤖 Modèle : {OLLAMA_MODEL}\n")

    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        print("❌ Ollama injoignable.")
        sys.exit(1)

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

        if _traiter_fichier(chemin, dossier_out, debug=args.debug):
            nb_ok += 1
        else:
            nb_ko += 1

    print(f"\n📊 Bilan : {nb_ok} OK, {nb_skip} ignorés, {nb_ko} erreurs")
    sys.exit(0 if nb_ko == 0 else 1)


if __name__ == "__main__":
    main()
