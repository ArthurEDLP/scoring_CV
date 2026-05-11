"""
Pré-traitement des AO brutes vers le format structuré attendu par le pipeline de scoring.

Utilise Ollama en local avec le modèle qwen2.5:7b-instruct.

Format d'entrée (AO brute) :
    {
      "id": "SNCF",
      "Profil": ["...", "..."],
      "Description": ["..."],
      "Contexte": ["..."]
    }

Format de sortie (AO structurée) :
    {
      "id": "SNCF",
      "entreprise": "SNCF",
      "poste": "Data Analyst / Data Scientist",
      "technos": ["Python", "SQL", ...],
      "seniorite_min_annees": 5,
      "Profil": [...],          # champs d'origine préservés
      "Description": [...],
      "Contexte": [...]
    }

Usage CLI :
    python pretraiter_ao.py AO_JSON_brutes/SNCF.json
        → écrit AO_JSON/SNCF.json

    python pretraiter_ao.py AO_JSON_brutes/*.json
        → traite tout le dossier

    python pretraiter_ao.py AO_JSON_brutes/SNCF.json --output mon_dossier/
        → choix du dossier de sortie

Pré-requis :
    - Ollama installé et lancé (`ollama serve`)
    - Modèle téléchargé : `ollama pull qwen2.5:7b-instruct`
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import requests   # pip install requests


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5:7b-instruct"
TIMEOUT_SEC   = 180         # extraction = quelques secondes en général
MAX_RETRIES   = 1           # si Qwen sort du JSON cassé, on retente


# ─────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'extraction d'informations \
depuis des offres d'emploi (AO). Tu réponds UNIQUEMENT avec du JSON valide, \
sans texte avant ou après, sans bloc markdown.

Tu extrais 4 champs depuis une AO :
  - "entreprise"          : nom de l'entreprise/client (string)
  - "poste"               : intitulé du poste recherché (string)
  - "technos"             : liste des technologies/outils explicitement \
mentionnés (array de strings)
  - "seniorite_min_annees": nombre d'années d'expérience minimum requis \
(integer, 0 si non précisé)

RÈGLES STRICTES :
1. N'INVENTE RIEN. Si une techno n'est pas explicitement citée, ne l'ajoute pas.
2. Garde l'orthographe d'origine pour les technos (ex: "AWS", "Snowflake").
3. Si la séniorité n'est pas précisée, mets 0 (pas null).
4. Si tu hésites sur l'entreprise, utilise l'id fourni.
5. Le poste doit être un intitulé de métier, pas une mission \
(ex: "Data Scientist" et non "construire des modèles ML").
6. Ne mets PAS les soft skills dans technos (ex: "rigueur", "autonomie" \
ne sont PAS des technos).
"""

EXEMPLE_FEW_SHOT = """EXEMPLE :

Input :
{
  "id": "TEST_BANQUE",
  "Profil": [
    "Vous maîtrisez Java, Spring Boot et Kafka",
    "Une expérience avec Docker et Kubernetes est un plus",
    "Vous justifiez de 3 ans minimum sur un poste similaire"
  ],
  "Description": ["Développer des microservices pour la plateforme bancaire"]
}

Output :
{"entreprise":"TEST_BANQUE","poste":"Développeur Backend","technos":["Java","Spring Boot","Kafka","Docker","Kubernetes"],"seniorite_min_annees":3}
"""


def _construire_prompt(ao_brute: Dict) -> str:
    """Construit le prompt complet à envoyer à Ollama."""
    # On retire les champs lourds inutiles avant envoi
    payload = {
        k: v for k, v in ao_brute.items()
        if k in ("id", "Profil", "Description", "Contexte")
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
    """
    Appelle l'API Ollama et retourne la réponse brute (string).
    Lève une exception en cas d'erreur réseau / serveur.
    """
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,        # on veut la réponse en un bloc
        "format":  "json",       # contrainte de format JSON côté Ollama
        "options": {
            "temperature": 0.1,  # peu de créativité, on veut du déterministe
        },
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Impossible de joindre Ollama sur localhost:11434.\n"
            "Vérifie qu'Ollama tourne (`ollama serve`)."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Erreur HTTP Ollama : {e}")


def _parser_reponse(reponse_brute: str) -> Dict:
    """
    Parse la réponse JSON, en tolérant un éventuel bloc markdown
    ```json ... ``` que Qwen pourrait ajouter malgré nos consignes.
    """
    txt = reponse_brute.strip()

    # Retire fences markdown si Qwen en a mis
    if txt.startswith("```"):
        # Retire les triples backticks et un éventuel "json" qui suit
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
    Vérifie que les 4 champs sont présents et bien typés.
    Retourne None si OK, sinon un message d'erreur.
    """
    champs_requis = {
        "entreprise":           str,
        "poste":                str,
        "technos":              list,
        "seniorite_min_annees": int,
    }
    for champ, type_attendu in champs_requis.items():
        if champ not in extrait:
            return f"champ manquant : '{champ}'"
        # On tolère int reçu comme float (ex: 5.0)
        if champ == "seniorite_min_annees" and isinstance(extrait[champ], float):
            extrait[champ] = int(extrait[champ])
        if not isinstance(extrait[champ], type_attendu):
            return (
                f"champ '{champ}' a le mauvais type "
                f"({type(extrait[champ]).__name__}, attendu {type_attendu.__name__})"
            )

    # Toutes les technos doivent être des strings non vides
    for i, t in enumerate(extrait["technos"]):
        if not isinstance(t, str) or not t.strip():
            return f"techno #{i} invalide : {t!r}"

    return None


# ─────────────────────────────────────────────────────────────────────
# Fonction principale (réutilisable depuis d'autres modules)
# ─────────────────────────────────────────────────────────────────────

def structurer_ao(ao_brute: Dict) -> Dict:
    """
    Prend une AO brute (format actuel avec Profil/Description/Contexte)
    et retourne sa version structurée enrichie des nouveaux champs.

    Les champs d'origine sont préservés, on ne fait qu'ajouter par-dessus.

    Args:
        ao_brute : dict de l'AO d'origine, doit avoir au moins un "id"

    Returns:
        dict enrichi avec entreprise/poste/technos/seniorite_min_annees

    Raises:
        RuntimeError : si l'extraction échoue après MAX_RETRIES tentatives
    """
    if "id" not in ao_brute:
        raise ValueError("L'AO doit avoir un champ 'id'")

    prompt = _construire_prompt(ao_brute)
    derniere_erreur = None

    for tentative in range(1, MAX_RETRIES + 2):  # 1 essai + N retries
        try:
            reponse_brute = _appeler_ollama(prompt)
            extrait       = _parser_reponse(reponse_brute)
            erreur        = _valider_extraction(extrait)

            if erreur:
                derniere_erreur = erreur
                continue  # retry

            # Succès : on fusionne avec l'AO d'origine
            ao_structuree = dict(ao_brute)        # copie shallow
            ao_structuree.update({
                "entreprise":           extrait["entreprise"],
                "poste":                extrait["poste"],
                "technos":              extrait["technos"],
                "seniorite_min_annees": extrait["seniorite_min_annees"],
            })
            return ao_structuree

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
    """Traite un fichier AO. Retourne True si succès."""
    try:
        with open(chemin_in, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ {chemin_in.name} : impossible de lire le JSON ({e})")
        return False

    # Tolère le format [{"data": {...}}] vu dans CV_AO_Loader
    if isinstance(data, list):
        data = data[0]
    ao_brute = data.get("data", data)

    print(f"  → {chemin_in.name} ... ", end="", flush=True)
    try:
        ao_structuree = structurer_ao(ao_brute)
    except Exception as e:
        print(f"✗ {e}")
        return False

    chemin_out = dossier_out / chemin_in.name
    with open(chemin_out, "w", encoding="utf-8") as f:
        json.dump(ao_structuree, f, ensure_ascii=False, indent=2)

    nb_technos = len(ao_structuree["technos"])
    seniorite  = ao_structuree["seniorite_min_annees"]
    print(
        f"✓ poste='{ao_structuree['poste']}', "
        f"{nb_technos} technos, {seniorite} ans"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Structure des AO brutes via Ollama (Qwen 2.5)."
    )
    parser.add_argument(
        "fichiers",
        nargs="+",
        help="Fichier(s) AO à traiter (supporte les wildcards shell, "
             "ex: AO_brutes/*.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./AO_JSON",
        help="Dossier de sortie (créé si absent). Défaut : ./AO_JSON",
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

    # Test connexion Ollama avant de boucler
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        print("❌ Ollama injoignable. Lance `ollama serve` dans un autre terminal.")
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
