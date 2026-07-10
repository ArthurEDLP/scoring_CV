"""
Axe "global" : similarité CV complet ↔ AO complète (Qwen3-Embedding-8B).

Indicateur informatif (c'est le score_final) : cosinus brut + rang dans le pool. Place le profil idéal en tête là où le matching techno par techno
peut le rater faute de labels littéraux.

Le cosinus est PRÉCALCULÉ par AO (script precalcul_global.py) pour éviter de faire tourner le 8B dans le pipeline. Ce module fournit :
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

MODEL = "qwen3-embedding:8b"


# ─────────────────────── Construction des textes ──────────────────────────

def _as_text(v) -> str:
    """Accepte string ou liste (details parfois sous forme de liste dans le brut)."""
    if isinstance(v, list):
        return " ".join(_as_text(x) for x in v if x) # on parcrours les éléments de la liste, la récursion permet de traiter les lite imbriquée et les éléments "vides"
    return (v or "").strip() if isinstance(v, str) else "" # si c'est un string

def texte_experience(exp: dict) -> str:
    """
    Créer des phrases pour contextualisé les éléments et améliorer l'embedding lors de la mise en commun des expériences
    """
    poste      = _as_text(exp.get("poste"))
    entreprise = _as_text(exp.get("entreprise"))
    details    = _as_text(exp.get("details"))
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


def texte_ao_complet(ao_data: Dict) -> str:
    """Profil + Description + Contexte, concaténés."""
    return " ".join(t for t in (
        _texte_section_ao(ao_data, "Profil"),
        _texte_section_ao(ao_data, "Description"),
        _texte_section_ao(ao_data, "Contexte"),
    ) if t)


# ─────────────────────── Embedding & cosinus (précalcul) ──────────────────

def _normaliser(vecteurs: np.ndarray) -> np.ndarray:
    """
    Normalisation L2, ligne par ligne. L2 c'est la norme euclidienne.
    Accepte (n, d) matrice de n vecteur d'embedding ou (d,). vecteur simple de dimension d on utilise Qwen3-embedding-8B donc d = 4096
    Remplace les normes nulles par 1.0 pour éviter NaN (division par 0).
    On divise les vecteurs par leurs normes pour les normaliser.

    Cette fonction nous permet d'obtenir directement la similarité cosinus entre deux embeddings.
    """
    arr = np.asarray(vecteurs, dtype="float32") # vérification du format du vecteur d'embedding
    if arr.ndim == 1:                           # on cherche à savoir si c'est un vecteur ou une matrice (un seul embedding ou pls), == 1 veut dire une vecteur (d, )
        n = np.linalg.norm(arr)                 # calcul de la norme (on utilise linalg pour linear algebra une bnaque de fonction mathématiques pour les vecteurs et matrices)
        return arr if n == 0 else arr / n
    norms = np.linalg.norm(arr, axis=1, keepdims=True) # le axis=1 car on calcul la norme horizontalement étant donné que chacune des lignes est un vecteur
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms

def embed(textes: List[str], model_name: str = MODEL) -> np.ndarray:
    """
    float32 est suffisant car il garde jusqu'à, environ, 7 chiffres significatifs (float 64 en prend 15) on économise notre processeur
    """
    import ollama # au cas où il y a un oublie dans le fichier où la fonction sera
    textes = [t[:10000] for t in textes] # garde fou pour des fichiers trop gros
    rep = ollama.embed(
        model=model_name,
        input=textes,
        keep_alive=-1) # le modèle reste en RAM pour accélèrer les appels suivants
    return np.array(rep["embeddings"], dtype=np.float32) # de l'appel on ne garde que l'embeddding et on passe la liste Python en un tableau NumPy


def embed_ollama(model_name: str, texte: str) -> np.ndarray:
    """Compat : un seul texte, normalisé."""
    return _normaliser(embed([texte], model_name)[0])


def cosinus_brut(emb_cv_complet: np.ndarray, emb_ao_complet: np.ndarray) -> float:
    if emb_cv_complet.size == 0 or emb_ao_complet.size == 0:
        return 0.0
    if emb_cv_complet.shape != emb_ao_complet.shape:
        raise ValueError("Les embeddings doivent avoir la même dimension.")
    return float(np.dot(emb_cv_complet, emb_ao_complet))


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
