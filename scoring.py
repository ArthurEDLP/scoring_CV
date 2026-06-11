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

from taxonomie import compatibles_technos
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
    return re.sub(r"\s+", " ", label.strip().lower())

def _tokens(label: str) -> set:
    # decoupe sur la ponctuation, ignore les tokens d'une lettre (R, C...)
    return {t for t in re.split(r"[^a-z0-9]+", label.lower()) if len(t) >= 2}


def score_technos(
    technos_ao,                       # List[{"label": str, "embedding": np.ndarray}]
    technos_cv,                       # idem
    seuil_semantique: float = 0.70,   # a calibrer 
    discount: float = 1.0,            # 1.0 = comportement actuel ; <1 pour devaluer le semantique
):
    if not technos_ao:
        return 1.0, {}

    n_ao = len(technos_ao)
    details = [None] * n_ao
    scores  = [0.0]  * n_ao

    # Index CV : norme + tokens + label original
    cv_norme  = [_normaliser(t["label"]) for t in technos_cv]
    cv_tokens = [_tokens(t["label"])      for t in technos_cv]
    cv_label  = [t["label"]               for t in technos_cv]
    cv_dispo  = set(range(len(technos_cv)))   # technos CV pas encore consommees

    # ── 1. EXACT (egalite stricte OU inclusion de tokens) ──────────────────
    # Un match exact CONSOMME la techno CV (1-a-1 des le depart).
    for a, t in enumerate(technos_ao):
        norme_ao  = _normaliser(t["label"])
        tokens_ao = _tokens(t["label"])

        # candidats parmi les CV encore disponibles
        cand = None
        for c in cv_dispo:
            if norme_ao == cv_norme[c]:                       # egalite stricte
                cand = c
                break
            if tokens_ao and tokens_ao <= cv_tokens[c]:       # AO ⊆ CV
                # on prefere le superset le plus serre
                if cand is None or len(cv_tokens[c]) < len(cv_tokens[cand]):
                    cand = c

        if cand is not None:
            cv_dispo.discard(cand)
            details[a] = {"score": 1.0, "source": "exact",
                          "matche_avec": cv_label[cand]}
            scores[a]  = 1.0

    # ── 2. SEMANTIQUE, affectation gloutonne 1-a-1 sur le reste ────────────
    ao_restantes = [a for a in range(n_ao) if details[a] is None]

    if ao_restantes and cv_dispo and technos_cv:
        mat_cv = np.stack([technos_cv[c]["embedding"] for c in sorted(cv_dispo)])
        cv_idx = sorted(cv_dispo)   # pour remapper ligne matrice -> index CV reel

        # matrice cosinus [AO_restantes x CV_dispo]
        couples = []
        for a in ao_restantes:
            emb_ao = technos_ao[a].get("embedding")
            if emb_ao is None:
                continue
            sims = mat_cv @ emb_ao
            for j, c in enumerate(cv_idx):
                if not compatibles_technos(technos_ao[a]["label"], cv_label[c]):
                    continue                     # familles incompatibles -> jamais candidat
                couples.append((float(sims[j]), a, c))

        # du meilleur cosinus au pire ; chaque AO et chaque CV servis une fois
        couples.sort(reverse=True)
        ao_pris, cv_pris = set(), set()
        for cos, a, c in couples:
            if cos < seuil_semantique:
                break
            if a in ao_pris or c in cv_pris:
                continue
            cos = float(np.clip(cos, 0.0, 1.0))
            details[a] = {"score": round(cos * discount, 3), "source": "semantique",
                          "matche_avec": cv_label[c], "cosine": round(cos, 3)}
            scores[a]  = cos * discount
            ao_pris.add(a); cv_pris.add(c)

    # ── 3. ABSENT ──────────────────────────────────────────────────────────
    for a, t in enumerate(technos_ao):
        if details[a] is None:
            details[a] = {"score": 0.0, "source": "absent", "matche_avec": None}
            scores[a]  = 0.0

    details_dict = {technos_ao[a]["label"]: details[a] for a in range(n_ao)}
    return float(np.mean(scores)), details_dict


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


def _norm_entreprise(s: str) -> str:
    """Minuscules, on garde le nom avant la 1re virgule (sans la ville), espaces compactés."""
    s = (s or "").split(",")[0]
    return re.sub(r"\s+", " ", s.strip().lower())

def _tokens_entreprise(s: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", s) if len(t) >= 2}


def score_bonus_entreprise(
    entreprise_ao: str,
    experiences_cv: List[Dict],
) -> Tuple[float, bool]:
    """Bonus si une expérience du CV est dans la même entreprise que l'AO.
    Tolérant à la ville en suffixe ('CANAL+, Nanterre') et à la casse."""
    if not entreprise_ao or not experiences_cv:
        return 0.0, False

    ao = _norm_entreprise(entreprise_ao)
    ta = _tokens_entreprise(ao)
    if not ao:
        return 0.0, False

    for exp in experiences_cv:
        cv = _norm_entreprise(exp.get("entreprise", ""))
        if not cv:
            continue
        tc = _tokens_entreprise(cv)
        if ao == cv or (ta and (ta <= tc or tc <= ta)):
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
