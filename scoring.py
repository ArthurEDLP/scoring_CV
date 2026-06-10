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
import re

# ────────────────────── Constantes ───────────────────────────


POIDS_AXES = {
    "technos":   0.70,
    "seniorite": 0.30,
    "bonus":     0.10,    # additif, peut faire dépasser 1.0 (ne le fera pas)
}

# Seuil pour qu'une expérience soit considérée "du poste AO".
SEUIL_CATEGORIE = 0.775

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

def _normaliser(label: str) -> str:
    """lower + trim + espaces compactés. Gère AWS/aws, pas k8s/kubernetes."""
    return re.sub(r"\s+", " ", label.strip().lower())

def _tokens(label: str) -> set:
    return set(_normaliser(label).split())

def score_technos(
    technos_ao,                       # List[{"label": str, "embedding": np.ndarray}]
    technos_cv,                       # idem
    seuil_semantique: float = 0.65,   # HAUT, à calibrer sur mpnet
    discount: float = 0.85,           # un match sémantique vaut moins qu'un exact ; mets 1.0 si tu n'en veux pas
) -> Tuple[float, Dict]:
    """
    Pour chaque techno AO :
      1. Match exact (label normalisé) → score 1.0
      2. Match sémantique (cosine ≥ seuil) → score = cosine * discount
      3. Absent → score 0.0
    Retourne (score_moyen, details: {techno_AO: {"score", "source", "matche_avec", ...}}).
    """
    if not technos_ao:
        return 1.0, {}

    # Index exact : label normalisé -> label original du CV (pour l'affichage)
    cv_par_norme = {}
    for t in technos_cv:
        cv_par_norme.setdefault(_normaliser(t["label"]), t["label"])

    mat_cv    = np.stack([t["embedding"] for t in technos_cv]) if technos_cv else None
    labels_cv = [t["label"] for t in technos_cv]

    details, scores = {}, []
    for t in technos_ao:
        norme = _normaliser(t["label"])
        tokens_ao = _tokens(t["label"])

        # 1. Exact
        label_match = None
        for norme_cv, label_cv, _ in cv_index:           # égalité stricte
            if norme_ao == norme_cv:
                label_match = label_cv
                break

        if label_match is None:                          # inclusion : AO ⊆ CV ex: "airflow" (AO) ⊆ "apache airflow" (CV)
            candidats = [(len(tok_cv), label_cv)
                         for _, label_cv, tok_cv in cv_index
                         if tokens_ao and tokens_ao <= tok_cv]
            if candidats:
                label_match = min(candidats)[1]          # superset le plus serré

        if label_match is not None:
            details[t["label"]] = {"score": 1.0, "source": "exact",
                                   "matche_avec": label_match}
            scores.append(1.0)
            continue

        # 2. Sémantique (seulement si rien trouvé en exact)
        if mat_cv is not None and t.get("embedding") is not None:
            sims = mat_cv @ t["embedding"]
            idx  = int(np.argmax(sims))
            cos  = float(np.clip(sims[idx], 0.0, 1.0))
            if cos >= seuil_semantique:
                details[t["label"]] = {
                    "score":       round(cos * discount, 3),
                    "source":      "semantique",
                    "matche_avec": labels_cv[idx],
                    "cosine":      round(cos, 3),
                }
                scores.append(cos * discount)
                continue

        # 3. Absent
        details[t["label"]] = {"score": 0.0, "source": "absent", "matche_avec": None}
        scores.append(0.0)

    return float(np.mean(scores)), details


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
