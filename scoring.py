"""
Module de scoring CV/AO.

Logique :
  1. CATÉGORISATION : un CV va EXACTEMENT dans une catégorie.
       - Si au moins 1 expérience matche le poste AO (sim >= SEUIL_CATEGORIE)
         → groupe principal (nommé comme le poste AO)
       - Sinon → 1 seul groupe alternatif, nommé comme le poste de
         l'expérience la plus récente du CV (ordre = 0).

  2. SÉNIORITÉ : somme TOTALE des expériences du CV (peu importe le poste).
     Calculée une fois pour toutes dans embedding_cache.

  3. SCORING :
     score = w_technos × score_technos
           + w_seniorite × score_seniorite
           + w_bonus × bonus_entreprise
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import math


# ────────────────────── Constantes ───────────────────────────


POIDS_AXES = {
    "technos":   0.50,
    "seniorite": 0.50,
    "bonus":     0.10,    # additif, peut faire dépasser 1.0
}

# Seuil pour qu'une expérience soit considérée "du poste AO".
SEUIL_CATEGORIE = 0.7

VALEUR_BONUS_ENTREPRISE = 1.0


# ──────────────────── Catégorisation ─────────────────────────────────


def categoriser_cv(
    emb_poste_ao: np.ndarray,
    poste_ao_label: str,
    experiences_cv: List[Dict],
    seuil: float = SEUIL_CATEGORIE,
) -> Optional[Dict]:
    """
    Détermine la catégorie unique d'un CV.

    Returns:
      {
        "nom":          "<nom de la catégorie>",
        "est_poste_ao": True/False,
      }
      ou None si le CV n'a aucune expérience exploitable.

    Cas concrets :
      - CV avec exp Data Scientist + Data Analyst, AO Data Scientist
        → {"nom": "Data Scientist", "est_poste_ao": True}

      - CV avec exp Commercial + Stagiaire, AO Data Scientist
        → {"nom": "Commercial", "est_poste_ao": False}
          (poste de l'exp d'ordre 0 = la plus récente)
    """
    if not experiences_cv:
        return None

    if emb_poste_ao is not None:
        # Y a-t-il au moins une expérience qui matche le poste AO ?
        for exp in experiences_cv:
            sim = float(np.dot(emb_poste_ao, exp["embedding"]))
            if sim >= seuil:
                return {"nom": poste_ao_label, "est_poste_ao": True}

    # Aucun match → groupe alternatif = poste de l'exp la plus récente
    # (par convention : la première dans l'ordre du CV, ordre = 0)
    exp_recente = min(experiences_cv, key=lambda e: e["ordre"])
    return {"nom": exp_recente["poste"], "est_poste_ao": False}


# ────────────────────── Score Technos ─────────────────────────────


def score_technos(
    technos_ao: List[Dict],
    technos_cv: List[Dict],
    seuil_match: float = 0.25, # par défaut 0.25, mais on peut choisir de mettre un autre minimun
) -> Tuple[float, Dict[str, float]]:
    """
    Pour chaque techno AO → meilleure sim avec une techno CV → moyenne.
    Retourne (score, details: {techno_AO: meilleure_sim}).
    """
    if not technos_ao:
        return 1.0, {}

    if not technos_cv:
        return 0.0, {t["label"]: 0.0 for t in technos_ao}

    mat_ao = np.stack([t["embedding"] for t in technos_ao])
    mat_cv = np.stack([t["embedding"] for t in technos_cv])

    sims = mat_ao @ mat_cv.T
    max_par_techno_ao = np.clip(sims.max(axis=1), 0.0, 1.0)

    if seuil_match > 0:
        max_par_techno_ao = np.where(
            max_par_techno_ao >= seuil_match,
            max_par_techno_ao,
            0.0,
        )

    score = float(max_par_techno_ao.mean())
    details = {
        t["label"]: float(s)
        for t, s in zip(technos_ao, max_par_techno_ao)
    }
    return score, details


# ─────────────────── Score Séniorité ─────────────────────────


def score_seniorite(
    seniorite_totale: float,
    seniorite_min_annees: float,
    tolerance_junior: float = 0.4,   # strict côté sous-qualifié
    tolerance_senior: float = 0.8,   # tolérant côté sur-qualifié
) -> float:
    """
    Gaussienne asymétrique. Pénalise plus durement le manque d'expérience que le surplus.
    """
    if seniorite_min_annees is None or seniorite_min_annees <= 0:
        return 1.0
    if seniorite_totale < 0:
        return 0.0

    mu = seniorite_min_annees
    ecart = seniorite_totale - mu

    # Sigma différent selon qu'on est junior ou sénior
    sigma = mu * (tolerance_junior if ecart < 0 else tolerance_senior)

    score = math.exp(-(ecart ** 2) / (2 * sigma ** 2))
    return float(score)


# ────────────────── Bonus Entreprise ───────────────────────────


def score_bonus_entreprise(
    entreprise_ao: str,
    experiences_cv: List[Dict],
) -> Tuple[float, bool]:
    """Match exact (lowercase + trim). Retourne (bonus, match_trouvé)."""
    if not entreprise_ao or not experiences_cv:
        return 0.0, False

    cible = entreprise_ao.strip().lower()
    for exp in experiences_cv:
        if exp.get("entreprise", "").strip().lower() == cible:
            return VALEUR_BONUS_ENTREPRISE, True
    return 0.0, False


# ─────────────────── Agrégation ──────────────────────────────


def agreger_scores(
    s_technos:   float,
    s_seniorite: float,
    s_bonus:     float,
    poids:       Optional[Dict[str, float]] = None,
) -> float:
    """
    Combinaison linéaire des 3 axes.
    Score non borné : bonus peuvent dépasser 1.0.
    """
    p = poids or POIDS_AXES
    return (
        p["technos"]   * s_technos
        + p["seniorite"] * s_seniorite
        + p["bonus"]     * s_bonus
    )
