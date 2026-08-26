"""
Module de scoring CV/AO.

Logique :
  1. CATÉGORISATION : un CV va EXACTEMENT dans une catégorie.
       - Si au moins 1 expérience matche le poste AO et est récente de moins de 2ans (sim >= SEUIL_SEMANTIQUE)
         -> groupe principal 
       - Sinon -> 1 seul groupe alternatif, l'expérience la plus récente du CV (ordre = 0).

  2. SÉNIORITÉ : somme TOTALE des expériences du CV (peu importe le poste).
     Calculée une fois pour toutes dans embedding_cache.

  3. SCORING :
     score = cos(cv_complet, ao_complet)
"""

from taxonomie import compatibles_technos
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import unicodedata
import math
import re

# ────────────────────── Constantes ───────────────────────────

SEUIL_TECHNO = 0.80 # score min pour valider une techno en orange

VALEUR_BONUS_ENTREPRISE = 1.0

SEUIL_EXP = 0.50

# ── Catégorisation principal/alternatif (nouvelle logique) ──
SEUIL_PRINCIPAL = 0.70          # score_pertinence_cv mini pour entrer dans le principal
ANCIENNETE_MAX   = 2.0           # une exp ne compte que si finie il y a <= 2 ans
POIDS_SECTIONS   = {"profil": 0.20, "description": 0.40, "contexte": 0.40}

# ──────────────────── Catégorisation ─────────────────────────────────


def score_pertinence_cv(
    sections_ao: Dict,                 # {"profil": np.ndarray|None, "description":..., "contexte":...}
    experiences_cv: List[Dict],
    anciennete_max: float = ANCIENNETE_MAX,
) -> Optional[Dict]:
    """
    Pour chaque expérience RÉCENTE (anciennete non nulle et <= anciennete_max), calcule un cosinus pondéré contre les sections AO :
        somme( POIDS_SECTIONS[s] * cos(exp, section_s) )  
        sur les sections présentes, poids renormalisés sur les sections réellement disponibles.

    Retourne le dict de la MEILLEURE expérience récente :
        {"score", "cos_profil", "cos_description", "cos_contexte", "poste", "anciennete"}
    ou None si aucune expérience récente exploitable (-> le CV ne peut pas être principal).
    """
    if not sections_ao or not experiences_cv: # on vérif que le sections de l'AO et les exps du CV ne soient pas vides
        return None

    # ------ AO
    # sections présentes + poids renormalisés (si une section AO manque)
    presentes = {section: emb # clé : valeur, on oublie pas
                for section, emb in sections_ao.items()
                if emb is not None and section in POIDS_SECTIONS}
    if not presentes: # on vérifie qu'au moins une des sections présente soit valide et son embedding aussi
        return None
    total_poids = sum(POIDS_SECTIONS[section] for section in presentes)

    # ------ CV
    meilleur = None
    for exp in experiences_cv:
        anc = exp.get("anciennete") # durée depuis la fin de cette exp, si en cours alors = 0
        if anc is None or anc > anciennete_max:      # indatable OU trop ancienne -> ignorée
            continue
        emb_exp = np.asarray(exp["embedding"], dtype="float32")

        cos = {section: None for section in POIDS_SECTIONS} # on met chaque section à None
        score = 0.0
        for section, emb_section in presentes.items():
            c = float(np.dot(emb_exp, np.asarray(emb_section, dtype="float32"))) # on peut calculer directe car les embeddings seront normailsé
            cos[section] = round(c, 4)
            score += (POIDS_SECTIONS[section] / total_poids) * c # donne le score de l'exp qui est au dessus du seuil d'ancienneté demandé

        if meilleur is None or score > meilleur["score"]: # permet d'avoir le meilleur score parmi les expériences récentes
            meilleur = {
                "score":          round(score, 4),
                **{f"cos_{s}": cos.get(s) for s in POIDS_SECTIONS}, # permet d'intégrer les cos de chaque section d'un coup, pas mal si on change POIDS_SECTION
                "poste":          exp.get("poste", ""),
                "anciennete":     round(anc, 2),
            }
    return meilleur


def categoriser_cv(
    sections_ao: Dict,
    poste_ao_label: str,
    experiences_cv: List[Dict],
    seuil: float = SEUIL_PRINCIPAL,
    anciennete_max: float = ANCIENNETE_MAX,
) -> Optional[Dict]:
    """
    Détermine la catégorie unique d'un CV via score_pertinence_cv.

    - score_pertinence_cv >= seuil (sur une exp récente) -> groupe PRINCIPAL
    - sinon -> groupe alternatif

    - nom = poste de l'expérience la plus récente (ordre = 0) qui est, ou non, dans le groupe principal

    Returns un dict :
      {"nom", "est_principal": bool, "score_pertinence_cv": float,
       "cos_profil", "cos_description", "cos_contexte"}
    ou None si le CV n'a aucune expérience exploitable.
    """
    if not experiences_cv:
        return None

    pert = score_pertinence_cv(sections_ao, experiences_cv, anciennete_max)
    exp_recente = min(experiences_cv, key=lambda e: e["ordre"])

    if pert and pert["score"] >= seuil: # le premier pert sert à éviter de planter si "pert is None", le deuxième à savoir si le meilleur des score est >= au seuil
        return {
            "nom":                 exp_recente["poste"],
            "est_principal":       True,
            "score_pertinence_cv": pert["score"],
            **{f"cos_{s}": pert.get(f"cos_{s}") for s in POIDS_SECTIONS},
        }

    
    return {
        "nom":                 exp_recente["poste"],
        "est_principal":       False,
        "score_pertinence_cv": pert["score"] if pert else 0.0,
        **{f"cos_{s}": (pert.get(f"cos_{s}") if pert else None) for s in POIDS_SECTIONS},
    }

def _formater_duree(annees: float) -> str:
    """2.42 -> '2 ans et 5 mois' ; 0.42 -> '5 mois' ; 1.0 -> '1 an'."""
    if annees is None or annees <= 0:
        return ""
    total_mois = round(annees * 12)
    ans, mois = divmod(total_mois, 12)
    bouts = []
    if ans:
        bouts.append(f"{ans} an" + ("s" if ans > 1 else ""))
    if mois:
        bouts.append(f"{mois} mois")
    return " et ".join(bouts) if bouts else "moins d'un mois"


def top_experiences(emb_ao_complete, experiences_cv, seuil: float = SEUIL_EXP, k=3):
    """<=k expériences les plus proches de l'AO complète (cosinus >= seuil), triées par cosinus décroissant. Vecteurs supposés normalisés."""
    if emb_ao_complete is None or not experiences_cv:
        return []
    emb_ao = np.asarray(emb_ao_complete, dtype="float32")
    scored = []
    for exp in experiences_cv:
        emb = exp.get("embedding")
        if emb is None:
            continue
        cos = float(np.dot(emb_ao, np.asarray(emb, dtype="float32")))
        if cos < seuil:
            continue
        annees = float(exp.get("annees", 0) or 0)
        scored.append({
            "poste":     exp.get("poste", ""),
            "entreprise": exp.get("entreprise", ""),
            "cosine":    round(cos, 4),
            "annees":    round(annees, 2),
            "duree_txt": _formater_duree(annees),
        })
    scored.sort(key=lambda e: e["cosine"], reverse=True)
    return scored[:k]

# ────────────────────── Score Technos ─────────────────────────────

# Mots de NIVEAU / connecteurs : ils ne définissent pas la techno, on les ignore
# pour le matching exact (sinon "SQL avancé" ≠ "SQL").
_QUALIF = {
    "avance", "avancee", "avancees", "avances", "basique", "basiques",
    "notion", "notions", "intermediaire", "intermediaires",
    "expert", "experte", "experts", "courant", "courante", "maitrise",
    "debutant", "debutante", "confirme", "confirmee", "experimente",
    "fondamentaux", "simple", "simples", "niveau", "bon", "bonne", "tres",
    "de", "des", "du", "la", "le", "les", "et", "ou", "au", "aux",
}

def _sans_accents(s: str) -> str:
    """'avancé' -> 'avance', 'données' -> 'donnees' (sinon la regex les fragmente)."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def _normaliser_texte(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())

def _coeur(label: str) -> str:
    """Retire le contenu entre parenthèses : 'Python (pandas)' -> 'Python'."""
    return re.sub(r"\([^)]*\)", " ", label)

def _tokens(label: str) -> set:
    """Tokens sans accents, sans mots de niveau, longueur >= 2."""
    s = _sans_accents(label.lower())
    toks = {t for t in re.split(r"[^a-z0-9]+", s) if len(t) >= 2}
    return toks - _QUALIF

def score_technos(
    technos_ao,                               # List[{"label": str, "embedding": np.ndarray}]
    technos_cv,                               # idem
    seuil_semantique: float = SEUIL_TECHNO,    
    discount: float = 1.0,                    # 1.0 = comportement actuel ; <1 pour devaluer le semantique
):
    """
    Score global des technos AO vs CV, avec détails par techno AO.
    Il y a 3 paliers de technologies: Exact (CV=AO) - Sémantique (La techno, CV, est proche de celle demandé par l'AO) - Absente 
    """
    if not technos_ao:
        return 1.0, {}

    nbr_tech_ao = len(technos_ao)
    details = [None] * nbr_tech_ao
    scores  = [0.0]  * nbr_tech_ao

    # Index CV : norme + tokens + label original
    cv_norme  = [_normaliser_texte(t["label"]) for t in technos_cv]
    cv_tokens = [_tokens(t["label"])      for t in technos_cv]
    cv_label  = [t["label"]               for t in technos_cv]
    cv_dispo  = set(range(len(technos_cv)))   # technos CV pas encore matché avec celle de l'AO

    # ── 1. EXACT (egalite stricte OU inclusion de tokens) ──────────────────
    # Un match exact CONSOMME la techno CV (1-a-1 des le depart).
    for ind, tech in enumerate(technos_ao):
        norme_ao  = _normaliser_texte(_coeur(tech["label"]))   # cœur : ignore la parenthèse
        tokens_ao = _tokens(_coeur(tech["label"]))

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
            details[ind] = {"score": 1.0, "source": "exact",
                          "matche_avec": cv_label[cand]}
            scores[ind]  = 1.0

    # ── 2. SEMANTIQUE, si une techno est validé elle ne sera pas ré-utilisé ────────────
    ao_restantes = [ind for ind in range(nbr_tech_ao) if details[ind] is None]

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
            for ind, c in enumerate(cv_idx):
                if not compatibles_technos(technos_ao[a]["label"], cv_label[c]):
                    continue                     # familles incompatibles -> jamais candidat
                couples.append((float(sims[ind]), a, c))

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
    for ind, t in enumerate(technos_ao):
        if details[ind] is None:
            details[ind] = {"score": 0.0, "source": "absent", "matche_avec": None}
            scores[ind]  = 0.0

    details_dict = {technos_ao[ind]["label"]: details[ind] for ind in range(nbr_tech_ao)}
    return float(np.mean(scores)), details_dict


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


# ────────────────── Classification en Catégorie de la Durée d'une Éxpérience ───────────────────────────

def _classifier_duree_experience(annees: float, seuil_court_mois: float, seuil_valide_mois: float) -> str:
    """
    Classifie une expérience en l'une des 3 catégories : courte - intermediaire - valide.
    """
    mois = annees * 12

    if seuil_court_mois > seuil_valide_mois:
        raise ValueError("seuil_court_mois doit être <= seuil_valide_mois")

    if mois <= seuil_court_mois:
        return "courte"
    
    if mois <= seuil_valide_mois :
        return "intermediaire"
    
    return "valide"


def classifier_experiences(experiences: list, seuil_court_mois: float, seuil_valide_mois: float) -> list:
    """
    Chaque expérience réduite aux champs d'affichage + 'duree_cat' (on ne peut pas garder l'embedding sinon ça fait planter FastAPI)
    """
    resultat = []
    for exp in experiences:
        annees = float(exp.get("annees", 0) or 0)
        resultat.append({
            "ordre":      exp.get("ordre"),
            "poste":      exp.get("poste", ""),
            "entreprise": exp.get("entreprise", ""),
            "annees":     round(annees, 2),
            "duree_cat":  _classifier_duree_experience(annees, seuil_court_mois, seuil_valide_mois),
        })
    return resultat


def compter_par_categorie(experiences_classees: list) -> dict:
    """
    Compte à partir d'expériences DÉJÀ classées (clé 'duree_cat')
    """
    counts = Counter(e["duree_cat"] for e in experiences_classees)
    return {c: counts.get(c, 0) for c in ("courte", "intermediaire", "valide")}