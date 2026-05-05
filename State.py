"""
État partagé du graphe LangGraph pour le matching CV/AO avec groupes.

Architecture :
  - noeud_categoriser : pour chaque CV, détermine ses catégories
                        (principale = poste AO, ou alternatives par poste)
  - noeud_technos     : score techno (indépendant de la catégorie)
  - noeud_bonus       : bonus entreprise (indépendant de la catégorie)
  - noeud_agreger     : combine tout, un classement par catégorie

La séniorité n'a plus de noeud dédié : elle est calculée à l'agrégation,
car elle dépend des années déjà filtrées par categoriser_cv.
"""

from typing import TypedDict, List, Annotated, Optional, Dict
from operator import add


class CategorisationEntry(TypedDict):
    """
    Le résultat de la catégorisation pour un CV.
    'categorie' = {"nom": ..., "est_poste_ao": True/False} ou None
    si le CV n'a aucune expérience exploitable.
    """
    cv_id: str
    categorie: Optional[Dict]


class ScoreTechnosEntry(TypedDict):
    cv_id: str
    score: float
    details: Dict[str, float]


class BonusEntrepriseEntry(TypedDict):
    cv_id: str
    bonus: float
    entreprise_matchee: bool


class CVScoringState(TypedDict):
    # Entrées
    offre: Dict
    cvs: List[Dict]

    # Sorties par axe
    categorisations:  Annotated[List[CategorisationEntry],   add]
    scores_technos:   Annotated[List[ScoreTechnosEntry],     add]
    bonus_entreprise: Annotated[List[BonusEntrepriseEntry],  add]

    # Résultat final : dict {nom_categorie: liste classée}
    # La catégorie "principale" porte le nom du poste AO.
    # Les alternatives portent l'intitulé exact des postes des CVs.
    resultats_par_categorie: Optional[Dict[str, List[Dict]]]

    # Logs d'erreurs
    erreurs: Annotated[List[str], add]


def state_initial(offre: Dict, cvs: List[Dict]) -> CVScoringState:
    return CVScoringState(
        offre                   = offre,
        cvs                     = cvs,
        categorisations         = [],
        scores_technos          = [],
        bonus_entreprise        = [],
        resultats_par_categorie = None,
        erreurs                 = [],
    )
