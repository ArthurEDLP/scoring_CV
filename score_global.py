"""
Axe de score "global" : similarité CV complet ↔ AO complète (Qwen3-Embedding).

Complète les axes fins (technos, séniorité…) par un signal sémantique de haut
niveau : « ce CV, dans son ensemble, ressemble-t-il à cette offre dans son
ensemble ? ». C'est l'axe qui place le profil idéal (type CEU) en tête, là où
le matching techno par techno peut le rater faute de labels littéraux.

Modèle recommandé (établi sur le benchmark) : Qwen/Qwen3-Embedding-8B.
  - 8B sans instruction       : CEU n°1, gap +0.10
  - 8B [inst:parcours]        : CEU n°1, gap +0.12 (meilleure discrimination)
  - 4B : ÉCHOUE (MFU passe devant CEU). e5 : trop tassé. -> ne pas utiliser.

GÉNÉRALISATION : le cosinus brut dépend de l'AO (plancher variable). On ne code
donc aucune borne en dur : on NORMALISE le cosinus par rapport au pool de
candidats de l'AO courante, recalculé à chaque run. Changer d'AO ne demande
aucun reparamétrage.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


# ─────────────────────── Construction des textes ──────────────────────────
# (identiques au benchmark, pour que prod et benchmark mesurent la même chose)

def texte_experience(exp: Dict) -> str:
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


def texte_cv_complet(cv_data: Dict) -> str:
    """Expériences + competences_techniques + savoir-faire/être, concaténés."""
    morceaux: List[str] = []
    for exp in cv_data.get("experiences", []) or []:
        if isinstance(exp, dict):
            t = texte_experience(exp)
            if t.strip():
                morceaux.append(t)
    for techno in cv_data.get("competences_techniques", []) or []:
        if isinstance(techno, str) and techno.strip():
            morceaux.append(techno.strip())
    for cle in ("savoir_faire", "savoir_etre", "savoir_faire/savoir_etre"):
        for item in cv_data.get(cle, []) or []:
            if isinstance(item, str) and item.strip():
                morceaux.append(item.strip())
    return " ".join(morceaux)


def _texte_section_ao(ao_data: Dict, section: str) -> str:
    contenu = ao_data.get(section, []) or []
    if isinstance(contenu, list):
        return " ".join(p.strip() for p in contenu
                        if isinstance(p, str) and p.strip())
    if isinstance(contenu, str):
        return contenu.strip()
    return ""


def texte_ao_complete(ao_data: Dict) -> str:
    """Profil + Description + Contexte concaténés."""
    return " ".join(t for t in (
        _texte_section_ao(ao_data, "Profil"),
        _texte_section_ao(ao_data, "Description"),
        _texte_section_ao(ao_data, "Contexte"),
    ) if t)


# ─────────────────────── Embedding & cosinus ──────────────────────────────

def embedder(model, textes: List[str], instruction: Optional[str] = None) -> np.ndarray:
    """
    Encode des textes en vecteurs NORMALISÉS (donc dot == cosinus).
    `instruction` ne s'applique qu'au côté AO (asymétrie Qwen3) :
    préfixe "Instruct: <...>\\nQuery: ".
    """
    if not textes:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    prompt = f"Instruct: {instruction}\nQuery: " if instruction else None
    if prompt:
        return model.encode(textes, prompt=prompt, normalize_embeddings=True).astype("float32")
    return model.encode(textes, normalize_embeddings=True).astype("float32")


def cosinus_brut(emb_cv: np.ndarray, emb_ao: np.ndarray) -> float:
    """Cosinus entre CV complet et AO complète (vecteurs déjà normalisés)."""
    if emb_cv.size == 0 or emb_ao.size == 0:
        return 0.0
    return float(np.dot(emb_cv, emb_ao))


# ─────────────────────── Indicateur du score global ───────────────────────────

def indicateur_global(cos_par_cv: Dict[str, float]) -> Dict[str, Dict]:
    """
    Indicateur informatif (n'entre PAS dans le score_final).
    Cosinus brut Qwen3-8B (CV complet ↔ AO complète) + rang dans le pool.
    """
    n = len(cos_par_cv)
    classement = sorted(cos_par_cv, key=cos_par_cv.get, reverse=True)
    rang = {cv: i + 1 for i, cv in enumerate(classement)}
    return {
        cv: {"cosine": round(cos_par_cv[cv], 4), "rang": rang[cv], "sur": n}
        for cv in cos_par_cv
    }