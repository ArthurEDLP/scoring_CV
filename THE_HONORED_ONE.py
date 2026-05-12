"""
Pipeline complet de matching CV ↔ Offre via LangGraph.

Architecture :
    START
      ├──→ categoriser ─┐
      ├──→ technos     ─┤──→ agreger ──→ END
      └──→ bonus       ─┘

Chaque CV apparaît dans EXACTEMENT UNE catégorie :
  - groupe principal (poste de l'AO) si match ≥ 0.90 sur au moins 1 exp
  - sinon : groupe alternatif nommé comme son poste le plus récent
"""

from typing import Dict, List
from collections import defaultdict

from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer

from CV_AO_Loader import charger_cvs, charger_offres
from embedding_cache import CacheEmbeddingsCV, CacheEmbeddingsOffre
from State import CVScoringState, state_initial
import scoring


# ════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════

print("⚙️  Chargement du modèle d'embedding...")
MODEL = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("⚙️  Initialisation des caches...")
CACHE_CV    = CacheEmbeddingsCV(MODEL,    "./cache_cv")
CACHE_OFFRE = CacheEmbeddingsOffre(MODEL, "./cache_offre")

print("✅ Setup terminé.\n")


# ════════════════════════════════════════════════════════════════════
# NOEUDS
# ════════════════════════════════════════════════════════════════════

def noeud_categoriser(state: CVScoringState) -> Dict:
    """Pour chaque CV, détermine sa catégorie unique."""
    offre = state["offre"]
    cvs   = state["cvs"]

    erreurs: List[str] = []
    entries: List[Dict] = []

    try:
        offre_emb = CACHE_OFFRE.obtenir(offre)
    except Exception as e:
        return {"erreurs": [f"[categoriser] offre {offre['id']}: {e}"]}

    poste_obj      = offre_emb["poste"]
    emb_poste_ao   = poste_obj["embedding"] if poste_obj else None
    poste_ao_label = poste_obj["label"] if poste_obj else "?"

    for cv in cvs:
        try:
            cv_emb = CACHE_CV.obtenir(cv)
            categorie = scoring.categoriser_cv(
                emb_poste_ao, poste_ao_label, cv_emb["experiences"]
            )
            entries.append({"cv_id": cv["id"], "categorie": categorie})
        except Exception as e:
            erreurs.append(f"[categoriser] CV {cv.get('id','?')}: {e}")

    return {"categorisations": entries, "erreurs": erreurs}


def noeud_technos(state: CVScoringState) -> Dict:
    offre = state["offre"]
    cvs   = state["cvs"]

    erreurs: List[str] = []
    entries: List[Dict] = []

    try:
        offre_emb = CACHE_OFFRE.obtenir(offre)
    except Exception as e:
        return {"erreurs": [f"[technos] offre {offre['id']}: {e}"]}

    technos_ao = offre_emb["technos"]

    for cv in cvs:
        try:
            cv_emb = CACHE_CV.obtenir(cv)
            s, details = scoring.score_technos(technos_ao, cv_emb["technos"])
            entries.append({
                "cv_id":   cv["id"],
                "score":   s,
                "details": details,
            })
        except Exception as e:
            erreurs.append(f"[technos] CV {cv.get('id','?')}: {e}")

    return {"scores_technos": entries, "erreurs": erreurs}


def noeud_bonus(state: CVScoringState) -> Dict:
    offre = state["offre"]
    cvs   = state["cvs"]

    erreurs: List[str] = []
    entries: List[Dict] = []

    entreprise_ao = (offre["data"].get("entreprise") or "").strip()

    for cv in cvs:
        try:
            cv_emb = CACHE_CV.obtenir(cv)
            bonus, match = scoring.score_bonus_entreprise(
                entreprise_ao, cv_emb["experiences"]
            )
            entries.append({
                "cv_id":              cv["id"],
                "bonus":              bonus,
                "entreprise_matchee": match,
            })
        except Exception as e:
            erreurs.append(f"[bonus] CV {cv.get('id','?')}: {e}")

    return {"bonus_entreprise": entries, "erreurs": erreurs}


# ════════════════════════════════════════════════════════════════════
# AGRÉGATION
# ════════════════════════════════════════════════════════════════════

def noeud_agreger(state: CVScoringState) -> Dict:
    """
    Pour chaque CV (UNE catégorie maximum) :
      - calcule la séniorité à partir de la séniorité totale du CV
      - assemble technos + séniorité + bonus
      - dépose dans le groupe correspondant
    Trie chaque groupe par score décroissant.
    """
    offre = state["offre"]
    annees_requises = float(offre["data"].get("seniorite_min_annees") or 0)

    # Indexation des contributions
    cat_par_cv:     Dict[str, Dict] = {}
    technos_par_cv: Dict[str, Dict] = {}
    bonus_par_cv:   Dict[str, Dict] = {}

    for e in state["categorisations"]:
        cat_par_cv[e["cv_id"]] = e["categorie"]
    for e in state["scores_technos"]:
        technos_par_cv[e["cv_id"]] = e
    for e in state["bonus_entreprise"]:
        bonus_par_cv[e["cv_id"]] = e

    # Pour chaque CV, on a besoin de sa séniorité totale → relire le cache
    # (rapide, c'est dans le JSON déjà sur disque)
    seniorite_par_cv: Dict[str, float] = {}
    for cv in state["cvs"]:
        try:
            cv_emb = CACHE_CV.obtenir(cv)
            seniorite_par_cv[cv["id"]] = cv_emb["seniorite_totale"]
        except Exception:
            seniorite_par_cv[cv["id"]] = 0.0

    par_categorie: Dict[str, List[Dict]] = defaultdict(list)

    for cv_id, categorie in cat_par_cv.items():
        if categorie is None:
            continue   # CV sans expérience exploitable

        s_technos_obj = technos_par_cv.get(cv_id, {})
        s_technos     = s_technos_obj.get("score",   0.0)
        details_tech  = s_technos_obj.get("details", {})

        bonus_obj = bonus_par_cv.get(cv_id, {})
        s_bonus   = bonus_obj.get("bonus",              0.0)
        match_ent = bonus_obj.get("entreprise_matchee", False)

        seniorite_totale = seniorite_par_cv.get(cv_id, 0.0)
        s_seniorite = scoring.score_seniorite(seniorite_totale, annees_requises)

        score_final = scoring.agreger_scores(
            s_technos   = s_technos,
            s_seniorite = s_seniorite,
            s_bonus     = s_bonus,
        )

        par_categorie[categorie["nom"]].append({
            "cv_id":              cv_id,
            "offre_id":           offre["id"],
            "score_final":        round(score_final, 4),
            "score_technos":      round(s_technos,   3),
            "score_seniorite":    round(s_seniorite, 3),
            "score_bonus":        round(s_bonus,     3),
            "seniorite_totale":   round(seniorite_totale, 1),
            "annees_requises":    annees_requises,
            "entreprise_matchee": match_ent,
            "technos_details":    details_tech,
            "est_poste_ao":       categorie["est_poste_ao"],
        })

    # Tri stable décroissant par groupe
    for cat in par_categorie:
        par_categorie[cat].sort(key=lambda r: r["score_final"], reverse=True)

    return {"resultats_par_categorie": dict(par_categorie)}


# ════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU GRAPHE
# ════════════════════════════════════════════════════════════════════

def construire_graphe():
    workflow = StateGraph(CVScoringState)

    workflow.add_node("categoriser", noeud_categoriser)
    workflow.add_node("technos",     noeud_technos)
    workflow.add_node("bonus",       noeud_bonus)
    workflow.add_node("agreger",     noeud_agreger)

    for n in ["categoriser", "technos", "bonus"]:
        workflow.add_edge(START, n)
        workflow.add_edge(n, "agreger")

    workflow.add_edge("agreger", END)
    return workflow.compile()


# ════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ════════════════════════════════════════════════════════════════════

def _afficher_groupe(nom: str, classement: List[Dict], top_k: int = 10) -> None:
    print(f"  ▸ {nom}  ({len(classement)} CV{'s' if len(classement) > 1 else ''})")
    print("  " + "─" * 76)
    for i, r in enumerate(classement[:top_k], 1):
        flag = " 🏢" if r["entreprise_matchee"] else ""
        print(
            f"  {i:>2}. {r['cv_id']:<14} "
            f"score={r['score_final']:.3f}  "
            f"technos={r['score_technos']:.2f}  "
            f"séniorité={r['score_seniorite']:.2f} "
            f"({r['seniorite_totale']:.1f}/{r['annees_requises']:.0f} ans)"
            f"{flag}"
        )
        manquantes = [
            f"{t}={s:.2f}"
            for t, s in r["technos_details"].items()
            if s < 0.5
        ]
        if manquantes:
            print(f"      ⚠️  technos faibles : {', '.join(manquantes[:5])}"
                  + (" ..." if len(manquantes) > 5 else ""))
    if len(classement) > top_k:
        print(f"      ... et {len(classement) - top_k} autres CV(s)")
    print()


def afficher_resultats(
    resultats_par_categorie: Dict[str, List[Dict]],
    offre_id: str,
    poste_ao: str,
    top_k: int = 10,
) -> None:
    if not resultats_par_categorie:
        print("Aucun résultat.")
        return

    print(f"\n{'═' * 78}")
    print(f"  AO : {offre_id}  —  Poste demandé : {poste_ao}")
    print(f"{'═' * 78}\n")

    principal = {
        nom: cl for nom, cl in resultats_par_categorie.items()
        if cl and cl[0]["est_poste_ao"]
    }
    alternatifs = {
        nom: cl for nom, cl in resultats_par_categorie.items()
        if cl and not cl[0]["est_poste_ao"]
    }

    if principal:
        print(f"🏆 GROUPE PRINCIPAL\n")
        for nom, cl in principal.items():
            _afficher_groupe(nom, cl, top_k=top_k)
    else:
        print(f"🏆 GROUPE PRINCIPAL  (aucun CV avec ces qualifications exact)\n")

    if alternatifs:
        print(f"📂 PROFILS ALTERNATIFS\n")
        # Tri : groupes les plus peuplés en premier
        for nom, cl in sorted(alternatifs.items(), key=lambda kv: -len(kv[1])):
            _afficher_groupe(nom, cl, top_k=top_k)


# ════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("📂 Chargement des données...")
    cvs    = charger_cvs("./CV_JSON")
    offres = charger_offres("./AO_JSON/AO_PMU.json")
    print(f"   {len(cvs)} CVs et {len(offres)} offres chargés.\n")

    if not cvs or not offres:
        print("❌ Aucune donnée à traiter.")
        exit(1)

    print("🧱 Construction du graphe LangGraph...")
    graphe = construire_graphe()
    print("✅ Graphe prêt.\n")

    offre_cible = offres[0]
    poste_ao    = offre_cible["data"].get("poste", "?")
    print(f"🚀 Matching pour l'offre {offre_cible['id']} ({poste_ao})...")

    state_init  = state_initial(offre_cible, cvs)
    state_final = graphe.invoke(state_init)

    if state_final["erreurs"]:
        print("\n⚠️  Erreurs rencontrées :")
        for err in state_final["erreurs"]:
            print(f"   - {err}")

    afficher_resultats(
        state_final["resultats_par_categorie"] or {},
        offre_id = offre_cible["id"],
        poste_ao = poste_ao,
    )
