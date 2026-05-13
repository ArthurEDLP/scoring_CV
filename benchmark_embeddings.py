"""
Benchmark de modèles d'embedding pour le matching CV/AO.

Pour chaque CV face à UNE AO précise (chemin fourni en argument), calcule
8 scores de similarité cosinus :

  Granularité "par expérience" (1 score par exp du CV) :
    1. expérience vs Profil
    2. expérience vs Description
    3. expérience vs Contexte
    4. expérience vs AO complète

  Granularité "CV complet" (1 score global) :
    5. CV complet vs Profil
    6. CV complet vs Description
    7. CV complet vs Contexte
    8. CV complet vs AO complète

Produit un fichier JSON nommé   benchmark_<modele>_<aoId>.json
(le nom de l'AO est inclus pour ne pas écraser un benchmark précédent).

Le texte d'une expérience est :   "<poste> chez <entreprise> : <details>"
Le texte du CV complet est : tout le texte des expériences + technos + savoir-faire/être
Le texte d'une section AO est : la concaténation des phrases de cette section
Le texte de l'AO complète est : Profil + Description + Contexte concaténés

Usage :
    python benchmark_embeddings.py <modele> <chemin_ao>

Exemples :
    python benchmark_embeddings.py paraphrase-multilingual-mpnet-base-v2 AO_JSON/PMU.json
    python benchmark_embeddings.py intfloat/multilingual-e5-large AO_JSON/CANAL+.json
    python benchmark_embeddings.py sentence-transformers/all-MiniLM-L6-v2 AO_JSON/SNCF.json --cv ./CV_JSON --output ./bench
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from CV_AO_Loader import charger_cvs


# ─────────────────────────────────────────────────────────────────────
# Construction des textes
# ─────────────────────────────────────────────────────────────────────

def _texte_experience(exp: Dict) -> str:
    """
    Construit le texte d'une expérience : '<poste> chez <entreprise> : <details>'.
    Tolère les champs manquants.
    """
    poste      = (exp.get("poste") or "").strip()
    entreprise = (exp.get("entreprise") or "").strip()
    details    = (exp.get("details") or "").strip()

    if entreprise and details:
        return f"{poste} chez {entreprise} : {details}"
    if entreprise:
        return f"{poste} chez {entreprise}"
    if details:
        return f"{poste} : {details}"
    return poste


def _texte_cv_complet(cv_data: Dict) -> str:
    """
    Concatène tout le texte exploitable du CV :
      - expériences (poste + entreprise + details)
      - competences_techniques
      - savoir_faire et/ou savoir_etre (gère l'ancien format 'savoir_faire/savoir_etre')
    """
    morceaux: List[str] = []

    # Expériences
    for exp in cv_data.get("experiences", []) or []:
        if isinstance(exp, dict):
            t = _texte_experience(exp)
            if t.strip():
                morceaux.append(t)

    # Compétences techniques
    for techno in cv_data.get("competences_techniques", []) or []:
        if isinstance(techno, str) and techno.strip():
            morceaux.append(techno.strip())

    # Savoir-faire / savoir-être : on gère les 2 formats (séparés ou fusionnés)
    for cle in ("savoir_faire", "savoir_etre", "savoir_faire/savoir_etre"):
        for item in cv_data.get(cle, []) or []:
            if isinstance(item, str) and item.strip():
                morceaux.append(item.strip())

    return " ".join(morceaux)


def _texte_section_ao(ao_data: Dict, section: str) -> str:
    """
    Concatène les phrases d'une section AO (Profil, Description, Contexte).
    Retourne "" si la section est absente ou vide.
    """
    contenu = ao_data.get(section, []) or []
    if isinstance(contenu, list):
        phrases = [p.strip() for p in contenu if isinstance(p, str) and p.strip()]
        return " ".join(phrases)
    if isinstance(contenu, str):
        return contenu.strip()
    return ""


def _texte_ao_complete(ao_data: Dict) -> str:
    """Concatène Profil + Description + Contexte."""
    return " ".join(
        t for t in (
            _texte_section_ao(ao_data, "Profil"),
            _texte_section_ao(ao_data, "Description"),
            _texte_section_ao(ao_data, "Contexte"),
        ) if t
    )


# ─────────────────────────────────────────────────────────────────────
# Calcul des scores
# ─────────────────────────────────────────────────────────────────────

def _duree_annees(date_str: str) -> float:
    """Parse 'MM/YYYY - MM/YYYY' / 'MM/YYYY - Aujourd'hui' → durée en années."""
    if not date_str:
        return 0.0
    matches = re.findall(r"(\d{1,2})/(\d{4})", date_str)
    if not matches:
        return 0.0
    from datetime import datetime
    try:
        m1, y1 = matches[0]
        debut = datetime(int(y1), int(m1), 1)
    except ValueError:
        return 0.0
    if len(matches) >= 2:
        try:
            m2, y2 = matches[1]
            fin = datetime(int(y2), int(m2), 1)
        except ValueError:
            fin = datetime.now()
        return max(0.0, (fin - debut).days / 365.25)
    # Une seule date + mention "en cours" → jusqu'à maintenant
    if re.search(r"aujourd['’]?hui|present|présent|current|ongoing|en cours|now",
                 date_str, re.IGNORECASE):
        return max(0.0, (datetime.now() - debut).days / 365.25)
    return 0.25  # date unique sans mention → durée par défaut


def _encode(model: SentenceTransformer, textes: List[str]) -> np.ndarray:
    """Encode une liste de textes, retourne une matrice normalisée (N, dim)."""
    if not textes:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return model.encode(textes, normalize_embeddings=True).astype("float32")


def _cosinus(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosinus entre 2 vecteurs déjà normalisés. Retourne 0 si l'un est vide."""
    if v1.size == 0 or v2.size == 0:
        return 0.0
    return float(np.dot(v1, v2))


def benchmark_cv_vs_ao(
    model: SentenceTransformer,
    cv: Dict,
    ao: Dict,
) -> Dict:
    """
    Calcule les 8 scores pour un (CV, AO) donné avec le modèle fourni.

    Returns:
      {
        "experiences_vs_sections": {
          "vs_profil":      [{"poste":..., "annees":..., "score":...}, ...],
          "vs_description": [...],
          "vs_contexte":    [...],
          "vs_ao_complete": [...]
        },
        "cv_complet_vs_sections": {
          "vs_profil":      <float>,
          "vs_description": <float>,
          "vs_contexte":    <float>,
          "vs_ao_complete": <float>
        }
      }
    """
    cv_data = cv["data"]
    ao_data = ao["data"]

    # ─── Textes côté AO ───
    txt_profil      = _texte_section_ao(ao_data, "Profil")
    txt_description = _texte_section_ao(ao_data, "Description")
    txt_contexte    = _texte_section_ao(ao_data, "Contexte")
    txt_ao_complete = _texte_ao_complete(ao_data)

    # ─── Embeddings côté AO ───
    ao_textes = [txt_profil, txt_description, txt_contexte, txt_ao_complete]
    ao_embeds = _encode(model, ao_textes)
    emb_profil, emb_desc, emb_ctx, emb_ao = ao_embeds

    # ─── Côté CV ───
    experiences = [e for e in (cv_data.get("experiences") or []) if isinstance(e, dict)]
    exp_textes  = [_texte_experience(e) for e in experiences]
    exp_embeds  = _encode(model, exp_textes)

    txt_cv_complet = _texte_cv_complet(cv_data)
    cv_embed = _encode(model, [txt_cv_complet])[0] if txt_cv_complet else np.array([])

    # ─── Scores par expérience ───
    def _scores_pour_section(emb_section: np.ndarray) -> List[Dict]:
        resultats = []
        # Si la section AO est vide, scores = 0 partout
        if emb_section.size == 0:
            for exp in experiences:
                resultats.append({
                    "poste":  exp.get("poste", ""),
                    "annees": round(_duree_annees(exp.get("date", "")), 2),
                    "score":  0.0,
                })
            return resultats
        for exp, emb_exp in zip(experiences, exp_embeds):
            score = _cosinus(emb_exp, emb_section) if emb_exp.size else 0.0
            resultats.append({
                "poste":  exp.get("poste", ""),
                "annees": round(_duree_annees(exp.get("date", "")), 2),
                "score":  round(score, 4),
            })
        return resultats

    experiences_vs_sections = {
        "vs_profil":      _scores_pour_section(emb_profil),
        "vs_description": _scores_pour_section(emb_desc),
        "vs_contexte":    _scores_pour_section(emb_ctx),
        "vs_ao_complete": _scores_pour_section(emb_ao),
    }

    # ─── Scores CV complet vs sections AO ───
    cv_complet_vs_sections = {
        "vs_profil":      round(_cosinus(cv_embed, emb_profil), 4),
        "vs_description": round(_cosinus(cv_embed, emb_desc),   4),
        "vs_contexte":    round(_cosinus(cv_embed, emb_ctx),    4),
        "vs_ao_complete": round(_cosinus(cv_embed, emb_ao),     4),
    }

    return {
        "experiences_vs_sections": experiences_vs_sections,
        "cv_complet_vs_sections":  cv_complet_vs_sections,
    }


# ─────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────

def _nom_sortie_safe(nom_modele: str) -> str:
    """Transforme 'intfloat/multilingual-e5-large' → 'intfloat_multilingual-e5-large'."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", nom_modele)


def lancer_benchmark(
    nom_modele:   str,
    chemin_ao:    str,
    dossier_cv:   str,
    dossier_out:  str,
) -> Path:
    """
    Lance le benchmark pour un modèle donné sur UNE AO précise (chemin).
    Écrit le JSON, retourne le chemin du fichier produit.
    """
    print(f"📂 Chargement des données...")

    # CVs depuis le dossier
    cvs = charger_cvs(dossier_cv)
    if not cvs:
        raise RuntimeError(f"Aucun CV trouvé dans : {dossier_cv}")
    print(f"   {len(cvs)} CVs chargés depuis {dossier_cv}")

    # AO : chargée directement depuis le chemin fourni
    chemin_ao_path = Path(chemin_ao)
    if not chemin_ao_path.exists():
        raise RuntimeError(f"Fichier AO introuvable : {chemin_ao}")

    with open(chemin_ao_path, "r", encoding="utf-8") as f:
        data_ao = json.load(f)
    # Tolère le format [{"data": {...}}] ou {"data": {...}} ou directement le data
    if isinstance(data_ao, list):
        data_ao = data_ao[0]
    ao = {
        "id":     data_ao.get("data", data_ao).get("id", chemin_ao_path.stem),
        "source": str(chemin_ao_path),
        "data":   data_ao.get("data", data_ao),
    }
    print(f"   AO utilisée : {ao['id']}  ({chemin_ao_path.name})")

    print(f"\n🤖 Chargement du modèle : {nom_modele}")
    print(f"   (téléchargement automatique au 1er usage)")
    model = SentenceTransformer(nom_modele)
    dim = model.get_sentence_embedding_dimension()
    print(f"   ✓ modèle chargé (dim = {dim})\n")

    # Boucle CVs
    resultats: Dict[str, Dict] = {}
    for i, cv in enumerate(cvs, 1):
        print(f"   [{i:>3}/{len(cvs)}] {cv['id']:<20} ... ", end="", flush=True)
        try:
            scores = benchmark_cv_vs_ao(model, cv, ao)
            resultats[cv["id"]] = {
                "cv_id":  cv["id"],
                "scores": scores,
            }
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            resultats[cv["id"]] = {"cv_id": cv["id"], "erreur": str(e)}

    # Structure finale
    payload = {
        "modele":     nom_modele,
        "dimension":  dim,
        "ao": {
            "id":    ao["id"],
            "poste": ao["data"].get("poste", ""),
        },
        "resultats":  resultats,
    }

    # Écriture : nom inclut modèle + AO (pas d'écrasement entre AO différentes)
    Path(dossier_out).mkdir(parents=True, exist_ok=True)
    nom_fichier = (
        f"benchmark_{_nom_sortie_safe(nom_modele)}"
        f"_{_nom_sortie_safe(ao['id'])}.json"
    )
    chemin = Path(dossier_out) / nom_fichier
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Résultats écrits dans : {chemin.resolve()}")
    return chemin


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de modèles d'embedding pour le matching CV/AO."
    )
    parser.add_argument(
        "modele",
        help="Nom du modèle sentence-transformers (ex: "
             "'paraphrase-multilingual-mpnet-base-v2' ou "
             "'intfloat/multilingual-e5-large')",
    )
    parser.add_argument(
        "ao",
        help="Chemin vers le fichier JSON de l'AO à benchmarker "
             "(ex: AO_JSON/PMU.json)",
    )
    parser.add_argument("--cv", default="./CV_JSON",
                        help="Dossier des CV (défaut: ./CV_JSON)")
    parser.add_argument("--output", "-o", default="./benchmark",
                        help="Dossier de sortie (défaut: ./benchmark)")
    args = parser.parse_args()

    try:
        lancer_benchmark(
            nom_modele  = args.modele,
            chemin_ao   = args.ao,
            dossier_cv  = args.cv,
            dossier_out = args.output,
        )
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
