"""
Garde-fous pour matching CV / offre.

Système binaire : un CV est soit ACCEPTÉ (entre dans le classement),
soit REJETÉ avec un motif (écarté). Le score n'est jamais modifié.

Deux usages possibles :

1. API directe (pour benchmark, scripts ad-hoc) :
       result = evaluer_cv(profile_id, cv_complet, exp_max, experiences, ...)
   Nécessite cv_complet et exp_max (cosines globaux CV/offre).

2. API pipeline (pour intégration LangGraph) :
       result = appliquer_filtre_recence(cv_brut, cv_emb, emb_poste_ao, ...)
   Travaille directement à partir des données du cache d'embeddings.

Filtres implémentés :

Filtre 1 — Cohérence du CV (nécessite cv_complet + exp_max) :
    Détecte un CV où une expérience isolée tire le score artificiellement.
    Signal : (exp_max - cv_complet) > DELTA.
    → REJETE_INCOHERENCE

Filtre 2 — Récence du parcours pertinent (utilisable sans cv_complet) :
    Parcourt les expériences de la plus récente à la plus ancienne :
      - première exp "dans le domaine" trouvée :
          → si fin ≤ FENETRE mois : ACCEPTE
          → sinon : REJETE_PARCOURS_OBSOLETE
      - on s'arrête à la première exp dans le domaine.
    Aucune exp dans le domaine du tout → REJETE_AUCUNE_EXP_PERTINENTE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, Iterable
import re

import numpy as np


# Seuil de proximité au "noyau de pertinence" du CV.
# Une exp i est dans le domaine si cosine_i >= exp_max - DELTA.
# CALIBRATION : 0.04 a été établi sur Qwen3-8B. À RECALIBRER pour mpnet
# en regardant la distribution des écarts (exp_max - cosine_i) sur des CV
# notoirement cohérents.
DELTA = 0.04

# Fenêtre temporelle au-delà de laquelle une expérience pertinente est obsolète.
DEFAULT_WINDOW_MONTHS = 24


class Statut(str, Enum):
    ACCEPTE = "ACCEPTE"
    REJETE_INCOHERENCE = "REJETE_INCOHERENCE"
    REJETE_PARCOURS_OBSOLETE = "REJETE_PARCOURS_OBSOLETE"
    REJETE_AUCUNE_EXP_PERTINENTE = "REJETE_AUCUNE_EXP_PERTINENTE"
    INDETERMINE = "INDETERMINE"  # données manquantes (ex: dates non parsables)


@dataclass
class Experience:
    poste: str
    entreprise: str
    date_debut: Optional[date]
    date_fin: Optional[date]  # None = "Aujourd'hui"
    cosine: float

    def mois_depuis_fin(self, ref: date) -> int:
        if self.date_fin is None:
            return 0
        dy = ref.year - self.date_fin.year
        dm = ref.month - self.date_fin.month
        return max(0, dy * 12 + dm)

    def date_tri(self, ref: date) -> date:
        return self.date_fin if self.date_fin is not None else ref


@dataclass
class GuardResult:
    profile_id: str
    statut: Statut
    motif: str
    details: dict = field(default_factory=dict)
    score: Optional[float] = None  # cv_complet si dispo, sinon None

    @property
    def accepte(self) -> bool:
        return self.statut == Statut.ACCEPTE


# ═══════════════════ Parsing dates ═════════════════════════════════════


def parse_date_str(date_str: str) -> Optional[date]:
    if not isinstance(date_str, str):
        return None
    s = date_str.strip().lower()
    if s in ("aujourd'hui", "aujourdhui", "present", "présent", "now", ""):
        return None
    m = re.match(r"(\d{1,2})/(\d{4})", date_str.strip())
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return date(year, month, 1)
    m = re.match(r"(\d{4})-(\d{1,2})", date_str.strip())
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    m = re.match(r"^(\d{4})$", date_str.strip())
    if m:
        return date(int(m.group(1)), 1, 1)
    return None


def parse_periode(periode: str) -> tuple[Optional[date], Optional[date]]:
    if not isinstance(periode, str):
        return (None, None)
    parts = re.split(r"\s*[-–—]\s*", periode, maxsplit=1)
    if len(parts) != 2:
        return (parse_date_str(periode), None)
    return (parse_date_str(parts[0]), parse_date_str(parts[1]))


def fenetre_obsolescence_mois(seniorite_min_annees: Optional[float] = None) -> int:
    if not seniorite_min_annees:
        return DEFAULT_WINDOW_MONTHS
    return max(DEFAULT_WINDOW_MONTHS, int(seniorite_min_annees) * 6)


# ═══════════════════ Filtres ═══════════════════════════════════════════


def filtre_1_coherence(cv_complet: float, exp_max: float,
                       delta: float = DELTA) -> tuple[bool, float]:
    """passe=True si CV cohérent (écart exp_max - cv_complet ≤ delta)."""
    ecart = exp_max - cv_complet
    return (ecart <= delta, ecart)


def filtre_2_recence(experiences: list[Experience],
                     exp_max: float,
                     ref_date: date,
                     fenetre_mois: int = DEFAULT_WINDOW_MONTHS,
                     delta: float = DELTA) -> tuple[Statut, dict]:
    """Trier de la plus récente à la plus ancienne. Statuer sur la première
    expérience trouvée dans le domaine."""
    seuil = exp_max - delta
    exps_triees = sorted(experiences,
                         key=lambda e: e.date_tri(ref_date),
                         reverse=True)

    for exp in exps_triees:
        if exp.cosine < seuil:
            continue  # hors domaine, on passe

        mois = exp.mois_depuis_fin(ref_date)
        details = {
            "seuil_domaine": round(seuil, 4),
            "exp_pertinente": f"{exp.poste} @ {exp.entreprise}",
            "cosine_exp": round(exp.cosine, 4),
            "mois_depuis_fin": mois,
            "fenetre_mois": fenetre_mois,
        }
        if mois <= fenetre_mois:
            return (Statut.ACCEPTE, details)
        return (Statut.REJETE_PARCOURS_OBSOLETE, details)

    return (Statut.REJETE_AUCUNE_EXP_PERTINENTE, {
        "seuil_domaine": round(seuil, 4),
        "fenetre_mois": fenetre_mois,
    })


# ═══════════════════ API directe (benchmark) ═══════════════════════════


def evaluer_cv(profile_id: str,
               cv_complet: float,
               exp_max: float,
               experiences: list[Experience],
               ref_date: date,
               seniorite_min_annees: Optional[float] = None,
               delta: float = DELTA) -> GuardResult:
    """Applique filtre 1 puis filtre 2. Premier rejet = sortie."""
    passe_1, ecart = filtre_1_coherence(cv_complet, exp_max, delta)
    base_details = {"filtre_1_ecart": round(ecart, 4)}

    if not passe_1:
        return GuardResult(
            profile_id=profile_id,
            score=cv_complet,
            statut=Statut.REJETE_INCOHERENCE,
            motif=f"Écart exp_max - cv_complet = +{ecart:.4f} > {delta}",
            details=base_details,
        )

    fenetre = fenetre_obsolescence_mois(seniorite_min_annees)
    statut, details_2 = filtre_2_recence(experiences, exp_max, ref_date, fenetre, delta)
    base_details.update(details_2)

    motif = _motif_depuis_statut(statut, details_2, fenetre)
    return GuardResult(
        profile_id=profile_id,
        score=cv_complet,
        statut=statut,
        motif=motif,
        details=base_details,
    )


def _motif_depuis_statut(statut: Statut, details: dict, fenetre: int) -> str:
    if statut == Statut.ACCEPTE:
        return "OK"
    if statut == Statut.REJETE_PARCOURS_OBSOLETE:
        return (f"Dernière expérience pertinente il y a "
                f"{details.get('mois_depuis_fin', '?')} mois "
                f"(fenêtre tolérée : {fenetre} mois)")
    if statut == Statut.REJETE_AUCUNE_EXP_PERTINENTE:
        return "Aucune expérience dans le domaine du poste"
    return "?"


# ═══════════════════ API pipeline (LangGraph) ══════════════════════════


def appliquer_filtre_recence(
    profile_id: str,
    cv_brut: dict,
    cv_emb: dict,
    emb_poste_ao: Optional[np.ndarray],
    ref_date: date,
    seniorite_min_annees: Optional[float] = None,
    delta: float = DELTA,
    fenetre_mois: Optional[int] = None,
) -> GuardResult:
    """
    Variante destinée au pipeline LangGraph :
    construit les Experience à partir du CV brut + du cache d'embeddings,
    puis applique le filtre 2 (récence). Le filtre 1 n'est PAS appliqué ici
    car il nécessite un cv_complet absent du pipeline.

    Args:
        profile_id: identifiant du CV (cv["id"])
        cv_brut: CV brut du loader (cv), pour les dates des expériences
        cv_emb: entrée du cache d'embeddings (CACHE_CV.obtenir(cv))
        emb_poste_ao: embedding du poste de l'offre. Si None → INDETERMINE.
        ref_date: date de référence pour le calcul d'obsolescence
        seniorite_min_annees: pour ajuster la fenêtre temporelle
        delta: marge de proximité au noyau
        fenetre_mois: override la fenêtre calculée depuis la séniorité
    """
    if emb_poste_ao is None:
        return GuardResult(
            profile_id=profile_id,
            statut=Statut.INDETERMINE,
            motif="Embedding du poste AO indisponible, filtre non applicable",
        )

    fenetre = fenetre_mois if fenetre_mois is not None \
              else fenetre_obsolescence_mois(seniorite_min_annees)

    # Construire les Experience en zippant cv_brut.experiences et cv_emb.experiences.
    # On suppose qu'elles sont dans le même ordre (= ordre du JSON brut).
    exps_brutes = (cv_brut.get("data") or cv_brut).get("experiences", [])
    exps_emb = cv_emb.get("experiences", [])

    if not exps_brutes or not exps_emb:
        return GuardResult(
            profile_id=profile_id,
            statut=Statut.REJETE_AUCUNE_EXP_PERTINENTE,
            motif="CV sans expérience exploitable",
        )

    # Alignement : on prend la plus courte des deux pour éviter les IndexError
    # (en pratique les deux listes ont la même longueur)
    experiences: list[Experience] = []
    for raw, emb in zip(exps_brutes, exps_emb):
        debut, fin = parse_periode(raw.get("date", ""))
        cosine = float(np.dot(emb_poste_ao, emb["embedding"]))
        experiences.append(Experience(
            poste=raw.get("poste", emb.get("poste", "?")),
            entreprise=raw.get("entreprise", emb.get("entreprise", "?")),
            date_debut=debut,
            date_fin=fin,
            cosine=cosine,
        ))

    exp_max = max(e.cosine for e in experiences)
    statut, details = filtre_2_recence(experiences, exp_max, ref_date, fenetre, delta)
    details["exp_max"] = round(exp_max, 4)

    motif = _motif_depuis_statut(statut, details, fenetre)
    return GuardResult(
        profile_id=profile_id,
        statut=statut,
        motif=motif,
        details=details,
    )
