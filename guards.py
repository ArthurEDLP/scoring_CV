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

class Disponibilite(str, Enum):
    """État de disponibilité d'un candidat, basé sur la fin de sa
    dernière expérience par rapport à la date du jour."""
    DISPO      = " 🔹 oui"          # dernière exp terminée (date passée)
    EN_CONTRAT = " 🔻 en contrat"   # dernière exp en cours ou fin future
    INCONNUE   = "?"            # impossible à déterminer
 
 
def detecter_disponibilite(
    experiences_brutes: list[dict],
    ref_date: date,
) -> Disponibilite:
    """
    Détermine la disponibilité d'un candidat à partir de sa dernière
    expérience.
 
    Logique :
      - Aucune expérience exploitable                       → INCONNUE
      - Dernière exp avec 'Aujourd'hui'/'present'/etc       → EN_CONTRAT
      - Dernière exp avec date_fin >= mois courant          → EN_CONTRAT
      - Dernière exp avec date_fin < mois courant           → DISPO
      - Date entièrement non parsable                       → INCONNUE
 
    Args:
        experiences_brutes: liste de dicts {date: "MM/YYYY - MM/YYYY", ...}
        ref_date: date de référence (typiquement date.today())
 
    Identification de la "dernière" expérience : on prend celle dont la
    date de fin est la plus récente (None = aujourd'hui = la plus récente).
    """
    if not experiences_brutes:
        return Disponibilite.INCONNUE
 
    exps_parsees = []
    for raw in experiences_brutes:
        date_str = raw.get("date", "")
        if not isinstance(date_str, str) or not date_str.strip():
            continue
 
        debut, fin = parse_periode(date_str)
        en_cours = _est_en_cours(date_str)
 
        # On garde l'exp si elle a au moins une info exploitable
        if debut is None and fin is None and not en_cours:
            continue
 
        exps_parsees.append({
            "debut":   debut,
            "fin":     fin,
            "en_cours": en_cours or (fin is None and debut is not None),
        })
 
    if not exps_parsees:
        return Disponibilite.INCONNUE
 
    # Trie : on prend l'expérience avec la date de fin la plus récente.
    # 'fin = None' (en cours) → on traite comme la date de référence
    # (donc la plus récente possible).
    derniere = max(
        exps_parsees,
        key=lambda e: e["fin"] if e["fin"] is not None else ref_date,
    )
 
    if derniere["en_cours"]:
        return Disponibilite.EN_CONTRAT
 
    fin = derniere["fin"]
    if fin is None:
        return Disponibilite.INCONNUE  # cas pathologique
 
    # Compare au mois près
    if (fin.year, fin.month) >= (ref_date.year, ref_date.month):
        return Disponibilite.EN_CONTRAT
    return Disponibilite.DISPO
 
 
def _est_en_cours(date_str: str) -> bool:
    """Détecte les mentions de fin en cours."""
    return bool(re.search(
        r"aujourd['’]?hui|present|présent|current|ongoing|en cours|now",
        date_str.lower(),
    ))

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
    
    infos = sorted(
    [(round(exp_max - e.cosine, 4), e.poste[:30], e.date_fin) for e in experiences]
    )
    print(f"[{exps_triees[0].poste[:20]}] exp_max={exp_max:.3f}")
    for ecart, poste, fin in infos:
        print(f"     écart={ecart:.4f}  {poste:<30} fin={fin}")

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
