"""
Axe "global" : similarité CV complet ↔ AO complète (Qwen3-Embedding-8B).

Indicateur informatif (n'entre PAS dans le score_final) : cosinus brut + rang
dans le pool. Place le profil idéal en tête là où le matching techno par techno
peut le rater faute de labels littéraux.

Le cosinus est PRÉCALCULÉ par AO (script precalcul_global.py) pour éviter de
faire tourner le 8B dans le pipeline. Ce module fournit :
  - les constructeurs de texte (identiques au benchmark),
  - embed_ollama / cosinus_brut (utilisés par le précalcul),
  - indicateur_global (cosinus + rang, utilisé à l'affichage),
  - chemin_cache_global (chemin partagé entre l'écriture et la lecture).
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List
import numpy as np


# ─────────────────────── Construction des textes ──────────────────────────

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


# ─────────────────────── Embedding & cosinus (précalcul) ──────────────────

def embed_ollama(model_name: str, texte: str) -> np.ndarray:
    """Embedding via ollama, NORMALISÉ (donc dot == cosinus)."""
    import ollama
    rep = ollama.embeddings(model=model_name, prompt=texte)
    v = np.asarray(rep["embedding"], dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosinus_brut(emb_cv: np.ndarray, emb_ao: np.ndarray) -> float:
    if emb_cv.size == 0 or emb_ao.size == 0:
        return 0.0
    return float(np.dot(emb_cv, emb_ao))


# ─────────────────────── Indicateur (affichage) ───────────────────────────

def indicateur_global(cos_par_cv: Dict[str, float]) -> Dict[str, Dict]:
    """
    Indicateur informatif : cosinus brut + rang dans le pool.
    N'entre PAS dans le score_final.
    """
    n = len(cos_par_cv)
    classement = sorted(cos_par_cv, key=cos_par_cv.get, reverse=True)
    rang = {cv: i + 1 for i, cv in enumerate(classement)}
    return {
        cv: {"cosine": round(cos_par_cv[cv], 4), "rang": rang[cv], "sur": n}
        for cv in cos_par_cv
    }


# ─────────────────────── Chemin de cache (partagé) ────────────────────────

def chemin_cache_global(ao_id: str, dossier: str = "./cache_global") -> Path:
    """Chemin du JSON de cosinus précalculés pour une AO. Utilisé en écriture
    (precalcul_global.py) ET en lecture (noeud_score_global)."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ao_id)
    return Path(dossier) / f"global_{safe}.json"
