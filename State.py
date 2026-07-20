"""
État partagé du graphe LangGraph pour le matching CV/AO avec groupes.
"""

from typing import TypedDict, List, Annotated, Optional, Dict
from operator import add

class ScoreGlobalEntry(TypedDict):
    cv_id: str
    cosine_brut: float


class CategorisationEntry(TypedDict):
    """
    Le résultat de la catégorisation pour un CV.
    'categorie' = {"nom": ..., "est_poste_ao": True/False} ou None si le CV n'a aucune expérience exploitable.
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


class CVRejeteEntry(TypedDict):
    """
    Un CV écarté par le nœud guards.
    Conserve toutes les infos de l'entry agrégée + statut + motif + détails + la catégorie d'où il a été retiré.
    """
    cv_id: str
    offre_id: str
    categorie: str
    score_technos: float
    score_bonus: float
    seniorite_totale: float
    annees_requises: float
    entreprise_matchee: bool
    technos_details: Dict[str, float]
    est_principal: bool
    score_pertinence_cv: Optional[float]
    cos_profil: Optional[float]
    cos_description: Optional[float]
    cos_contexte: Optional[float]
    guards_statut: str
    guards_motif: str
    guards_details: Dict


class CVScoringState(TypedDict):
    # Entrées
    offre: Dict
    cvs: List[Dict]

    # Sorties par axe
    categorisations:  Annotated[List[CategorisationEntry],   add]
    scores_technos:   Annotated[List[ScoreTechnosEntry],     add]
    bonus_entreprise: Annotated[List[BonusEntrepriseEntry],  add]

    # Sortie de noeud_agreger : classement brut, avant guards.
    # La catégorie "principale" porte l'intitulé exact du dernier poste du CV.
    # Les alternatives portent l'intitulé exact du dernier poste du CV.
    resultats_par_categorie: Optional[Dict[str, List[Dict]]]

    # Sorties de noeud_guards :
    #   - resultats_acceptes_par_categorie : même structure que resultats_par_categorie, mais purgé des CV rejetés. C'est ce qui est utilisé pour l'affichage final.
    #   - cv_rejetes : liste plate des CV écartés, avec leur motif.
    # Si noeud_guards n'a pas tourné (graphe sans cette étape), ces deux
    # champs restent à None et le code aval doit retomber sur
    # resultats_par_categorie.
    resultats_acceptes_par_categorie: Optional[Dict[str, List[Dict]]]
    cv_rejetes: Optional[List[CVRejeteEntry]]

    # Logs d'erreurs
    erreurs: Annotated[List[str], add]

    scores_globaux: Annotated[List[ScoreGlobalEntry], add]



def state_initial(offre: Dict, cvs: List[Dict]) -> CVScoringState:
    return CVScoringState(
        offre                            = offre,
        cvs                              = cvs,
        categorisations                  = [],
        scores_technos                   = [],
        bonus_entreprise                 = [],
        resultats_par_categorie          = None,
        resultats_acceptes_par_categorie = None,
        cv_rejetes                       = None,
        erreurs                          = [],
        scores_globaux                   = [],
    )
