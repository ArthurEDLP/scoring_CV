"""
Pipeline complet de matching CV ↔ Offre via LangGraph.

Architecture :

  - noeud_categoriser : pour chaque CV, détermine ses catégories (principale = poste AO, ou alternatives par poste)

  - noeud_technos     : score techno (indépendant de la catégorie)

  - noeud_bonus       : bonus entreprise (indépendant de la catégorie)

  - noeud_agreger     : combine tout, un classement par catégorie

  - noeud_guards      : filtre de récence sur le parcours, sépare les CV acceptés (entrent dans le classement) et rejetés (écartés avec motif)

La séniorité n'a plus de noeud dédié : elle est calculée à l'agrégation, car elle dépend des années déjà filtrées par categoriser_cv.

    START
      ├──> categoriser ─┐
      ├──> score global─┤
      ├──> technos     ─┤──> agreger ──> guards ──> END
      └──> bonus       ─┘

Chaque CV apparaît dans EXACTEMENT UNE catégorie :
  - groupe principal si match ≥ SEUIL_PRINCIPAL sur au moins 1 exp qui a eu lieu il y a moins de 2ans (cf scoring.py)
  - sinon : groupe alternatif

Le nœud `guards` applique ensuite un filtre de récence sur le parcours :
les CV dont l'expérience pertinente est trop ancienne sont écartés du classement et basculent dans une liste de rejetés (avec motif).
"""

from datetime import date
from typing import Dict, List
from collections import defaultdict
import json
import numpy as np

from langgraph.graph import StateGraph, START, END
import ollama

from CV_AO_Loader import charger_cvs, charger_offres
from embedding_cache import CacheEmbeddingsCV, CacheEmbeddingsOffre
from State import CVScoringState, state_initial
import scoring
import guards
from guards import detecter_disponibilite
from score_global import indicateur_global, chemin_cache_global


# ════════════════════════ SETUP ═════════════════════════════════════


print("Chargement du modèle d'embedding...")
MODEL = "qwen3-embedding:4b"

print("Initialisation des caches...")
CACHE_CV    = CacheEmbeddingsCV(MODEL,    "./cache_cv")
CACHE_OFFRE = CacheEmbeddingsOffre(MODEL, "./cache_offre")
DOSSIER_CACHE_GLOBAL = "./cache_global"   # cosinus précalculés par precalcul_global.py


print("Setup terminé.\n")


# Paramètres des garde-fous.
# DELTA : à recalibrer sur la distribution observée
GUARDS_DELTA = 0.5
GUARDS_FENETRE_MOIS = None  # None = dérivé de seniorite_min_annees (24 mois min)

# ═══════════════════════ NOEUDS ════════════════════════════════════


def noeud_categoriser(state: CVScoringState) -> Dict:
    """Pour chaque CV, détermine sa catégorie unique."""
    offre = state["offre"]
    cvs   = state["cvs"]
    seuils = state["config_seuils"]
    seuil_court  = seuils["seuil_court_mois"]
    seuil_valide = seuils["seuil_valide_mois"]

    erreurs: List[str] = []
    entries: List[Dict] = []

    try:
        offre_emb = CACHE_OFFRE.obtenir(offre) # on récupère les AO après embedding
    except Exception as e:
        return {"erreurs": [f"[categoriser] offre {offre['id']}: {e}"]}

    poste_obj      = offre_emb["poste"]
    poste_ao_label = poste_obj["label"] if poste_obj else "?"  # nom du poste
    sections_ao    = offre_emb.get("sections") or {}           # profil/description/contexte

    for cv in cvs:
        try:
            cv_emb = CACHE_CV.obtenir(cv)

            # ── Catégorisation Principal / Alternatif (utilise cv_emb, intact) ──
            categorie = scoring.categoriser_cv(
                sections_ao, poste_ao_label, cv_emb["experiences"]
            )

            # ── Classification des durées (nouvelle variable, pas d'écrasement) ──
            exps_durees = scoring.classifier_experiences(
                cv_emb["experiences"], seuil_court, seuil_valide
            )
            compteurs = scoring.compter_par_categorie(exps_durees)

            entries.append({
                "cv_id":               cv["id"],
                "categorie":           categorie,
                "experiences_durees":  exps_durees,   
                "compteurs_durees":    compteurs,      # {"courte": n, ...}
            })
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


def noeud_score_global(state: CVScoringState) -> Dict:
    """
    Indicateur : cosinus CV complet <-> AO complète (Qwen3-4B).
    NE FAIT PAS tourner le 4B : il lit le JSON précalculé par precalcul_global.py.
    Si le cache est absent, l'indicateur est simplement ignoré (erreur loguée).
    """
    ao_id  = state["offre"]["id"]
    chemin = chemin_cache_global(ao_id, DOSSIER_CACHE_GLOBAL)

    if not chemin.exists():
        return {"erreurs": [
            f"[score_global] cache absent pour l'AO {ao_id} ({chemin}). "
            f"Lance : python precalcul_global.py --only {ao_id}"
        ]}

    try:
        cosines = json.loads(chemin.read_text(encoding="utf-8")).get("cosines", {})
    except Exception as e:
        return {"erreurs": [f"[score_global] lecture cache {ao_id}: {e}"]}

    erreurs: List[str] = []
    sorties: List[Dict] = []
    for cv in state["cvs"]:
        c = cosines.get(cv["id"])
        if c is None:
            erreurs.append(f"[score_global] CV {cv['id']} absent du cache {ao_id} "
                           f"(relance precalcul_global.py)")
            continue
        sorties.append({"cv_id": cv["id"], "cosine_brut": float(c)})

    return {"scores_globaux": sorties, "erreurs": erreurs}

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


# ═════════════════════════ AGRÉGATION ════════════════════════════════


def noeud_agreger(state: CVScoringState) -> Dict:
    """
    Pour chaque CV (UNE catégorie maximum) :
      - calcule la séniorité à partir de la séniorité totale du CV
      - assemble technos + séniorité + bonus + score global
      - dépose dans le groupe correspondant
    Trie chaque groupe par score décroissant.
    """
    offre = state["offre"]
    annees_requises = float(offre["data"].get("seniorite_min_annees") or 0)

    cat_par_cv:     Dict[str, Dict] = {}
    technos_par_cv: Dict[str, Dict] = {}
    bonus_par_cv:   Dict[str, Dict] = {}
    durees_par_cv:    Dict[str, List] = {}
    compteurs_par_cv: Dict[str, Dict] = {}

    # ───────────  Transformation des listes en dictionnaire d'index  ────
    """
    L'ordre d'appartition des CVs n'est pas garanti le même DONC on doit effectuer une transformation pour faciliter l'appel des CVs

    On passe de se format :

    state["categorisations"] = [
    {"cv_id": "cv_B", "categorie": {...}},
    {"cv_id": "cv_A", "categorie": {...}},
    {"cv_id": "cv_C", "categorie": None},
    ]
    state["scores_technos"] = [
    {"cv_id": "cv_C", "score": 0.8, "details": {...}},
    {"cv_id": "cv_B", "score": 0.6, "details": {...}},
    {"cv_id": "cv_A", "score": 0.9, "details": {...}},
    ]

    à celui-ci:

    cat_par_cv = {
    "cv_A": {...},          # la valeur = e["categorie"] , "categorie" étant la sortie de categoriser_cv
    "cv_B": {...},
    "cv_C": None,
    }
    technos_par_cv = {
    "cv_A": {"cv_id": "cv_A", "score": 0.6, "details": {...}},   # la valeur = e (l'entry entière) , pas comme avant, ici on recopie le dictionnaire
    "cv_B": {"cv_id": "cv_B", "score": 0.8, "details": {...}},
    "cv_C": {"cv_id": "cv_C", "score": 0.9, "details": {...}},
    }
    """

    for e in state["categorisations"]:
        cat_par_cv[e["cv_id"]]       = e["categorie"]
        durees_par_cv[e["cv_id"]]    = e.get("experiences_durees", [])   
        compteurs_par_cv[e["cv_id"]] = e.get("compteurs_durees", {})
    for e in state["scores_technos"]:
        technos_par_cv[e["cv_id"]] = e
    for e in state["bonus_entreprise"]:
        bonus_par_cv[e["cv_id"]] = e

    seniorite_par_cv: Dict[str, float] = {}
    for cv in state["cvs"]:
        try:
            cv_emb = CACHE_CV.obtenir(cv)
            seniorite_par_cv[cv["id"]] = cv_emb["seniorite_totale"]
        except Exception:
            seniorite_par_cv[cv["id"]] = 0.0 # au lieu d'afficher un message d'erreur on met la seniorité à 0

    # Indicateur global (lecture seule, n'entre PAS dans score_final)
    cos_par_cv = {e["cv_id"]: e["cosine_brut"] for e in state["scores_globaux"]}
    indic = indicateur_global(cos_par_cv)

    # Embedding de l'AO complète : récupéré depuis le cache
    # (au lieu d'appeler Ollama à chaque matching)
    offre_emb = CACHE_OFFRE.obtenir(offre)
    ao_complete = offre_emb.get("ao_complete")
    if ao_complete is not None:
        emb_ao_complete = ao_complete["embedding"]
        print("[top_experiences] embedding AO chargé depuis cache")
    else:
        emb_ao_complete = None
        print("[top_experiences] AO complète absente du cache")

    par_categorie: Dict[str, List[Dict]] = defaultdict(list)

    for cv_id, categorie in cat_par_cv.items():
        if categorie is None:
            continue

        s_technos_obj = technos_par_cv.get(cv_id, {})
        s_technos     = s_technos_obj.get("score",   0.0)
        details_tech  = s_technos_obj.get("details", {})

        bonus_obj = bonus_par_cv.get(cv_id, {})
        s_bonus   = bonus_obj.get("bonus",              0.0)
        match_ent = bonus_obj.get("entreprise_matchee", False)

        seniorite_totale = seniorite_par_cv.get(cv_id, 0.0)

        # Récupère le CV brut pour parser ses dates (le CV brut est dans state["cvs"])
        cv_brut = next((c for c in state["cvs"] if c["id"] == cv_id), None)
        exps_brutes = (cv_brut.get("data") or cv_brut).get("experiences", []) if cv_brut else []
        dispo = detecter_disponibilite(exps_brutes, date.today())

        cv_emb_obj = CACHE_CV.obtenir(cv_brut) if cv_brut else {"experiences": []}
        tops = scoring.top_experiences(emb_ao_complete, cv_emb_obj.get("experiences", []), k=3)

        par_categorie[categorie["nom"]].append({
            "cv_id":              cv_id,
            "offre_id":           offre["id"],
            "score_technos":      round(s_technos,   3),
            "score_bonus":        round(s_bonus,     3),
            "seniorite_totale":   round(seniorite_totale, 1),
            "annees_requises":    annees_requises,
            "entreprise_matchee": match_ent,
            "technos_details":    details_tech,
            "est_principal":      categorie["est_principal"],
            "score_pertinence_cv": categorie.get("score_pertinence_cv"),
            "cos_profil":         categorie.get("cos_profil"),
            "cos_description":    categorie.get("cos_description"),
            "cos_contexte":       categorie.get("cos_contexte"),
            "disponibilite":      dispo.value,
            "indicateur_global":  indic.get(cv_id),
            "top_experiences":    tops,
            "experiences_durees": durees_par_cv.get(cv_id, []),     
            "compteurs_durees":   compteurs_par_cv.get(cv_id, {}),
        })

    # Tri de chaque groupe par l'indicateur global (cosinus CV complet <-> AO complète)
    def _cos(r):
        ig = r.get("indicateur_global")
        return ig["cosine"] if ig else -1.0
    for cat in par_categorie:
        par_categorie[cat].sort(key=_cos, reverse=True)

    return {"resultats_par_categorie": dict(par_categorie)}


# ═════════════════════════ GUARDS ════════════════════════════════════


def noeud_guards(state: CVScoringState) -> Dict:
    """
    Applique le filtre de récence sur le parcours :
    pour chaque CV déjà classé, on regarde si au moins une expérience "dans le domaine du poste AO" est récente (≤ fenêtre).

    Sortie :
      - resultats_acceptes_par_categorie : même structure que
        resultats_par_categorie, mais ne contient que les CV acceptés
      - cv_rejetes : liste des CV écartés avec leur motif

    Note : les rejetés ne sont plus dans le classement principal ; ils sont triés à titre informatif par l'affinité globale (indicateur_global.cosine).
    """
    offre = state["offre"]
    cvs_par_id = {cv["id"]: cv for cv in state["cvs"]}
    annees_requises = offre["data"].get("seniorite_min_annees")
    ref_date = date.today()

    erreurs: List[str] = []

    # Embedding du poste AO
    try:
        offre_emb = CACHE_OFFRE.obtenir(offre)
        poste_obj = offre_emb.get("poste")
        emb_poste_ao = poste_obj["embedding"] if poste_obj else None
    except Exception as e:
        erreurs.append(f"[guards] offre {offre['id']}: {e}")
        emb_poste_ao = None

    # Pour chaque CV : appliquer le filtre
    statuts: Dict[str, guards.GuardResult] = {}
    for cv in state["cvs"]:
        try:
            cv_emb = CACHE_CV.obtenir(cv)
            result = guards.appliquer_filtre_recence(
                profile_id=cv["id"],
                cv_brut=cv,
                cv_emb=cv_emb,
                emb_poste_ao=emb_poste_ao,
                ref_date=ref_date,
                seniorite_min_annees=annees_requises,
                delta=GUARDS_DELTA,
                fenetre_mois=GUARDS_FENETRE_MOIS,
            )
            statuts[cv["id"]] = result
        except Exception as e:
            erreurs.append(f"[guards] CV {cv.get('id','?')}: {e}")

    # Séparer les résultats agrégés en acceptés / rejetés
    resultats_acceptes: Dict[str, List[Dict]] = defaultdict(list)
    cv_rejetes: List[Dict] = []

    for cat_nom, classement in (state["resultats_par_categorie"] or {}).items():
        for entry in classement:
            cv_id = entry["cv_id"]
            result = statuts.get(cv_id)

            if result is None or result.statut == guards.Statut.INDETERMINE:
                # On ne peut pas statuer → on laisse passer mais on note
                entry_enrichi = {**entry, "guards_statut": "INDETERMINE"}
                resultats_acceptes[cat_nom].append(entry_enrichi)
                continue

            if result.accepte:
                resultats_acceptes[cat_nom].append({
                    **entry,
                    "guards_statut": result.statut.value,
                })
            else:
                cv_rejetes.append({
                    **entry,
                    "categorie": cat_nom,
                    "guards_statut": result.statut.value,
                    "guards_motif": result.motif,
                    "guards_details": result.details,
                })

    # Tri des rejetés par affinité globale décroissante (score_final supprimé)
    def _cos_rej(r):
        ig = r.get("indicateur_global")
        return ig["cosine"] if ig else -1.0
    cv_rejetes.sort(key=_cos_rej, reverse=True)

    return {
        "resultats_acceptes_par_categorie": dict(resultats_acceptes),
        "cv_rejetes": cv_rejetes,
        "erreurs": erreurs,
    }


# ═══════════════ CONSTRUCTION DU GRAPHE ════════════════════════


def construire_graphe():
    workflow = StateGraph(CVScoringState)

    workflow.add_node("categoriser", noeud_categoriser)
    workflow.add_node("technos",     noeud_technos)
    workflow.add_node("bonus",       noeud_bonus)
    workflow.add_node("agreger",     noeud_agreger)
    workflow.add_node("guards",      noeud_guards)
    workflow.add_node("score_global", noeud_score_global)

    for n in ["categoriser", "technos", "bonus", "score_global"]:
        workflow.add_edge(START, n)
        workflow.add_edge(n, "agreger")

    workflow.add_edge("agreger", "guards")
    workflow.add_edge("guards", END)
    return workflow.compile()


# ════════════════════════ AFFICHAGE ═══════════════════════════


def _afficher_groupe(nom: str, classement: List[Dict], top_k: int = 10) -> None:
    print(f"  ▸ {nom}  ({len(classement)} CV{'s' if len(classement) > 1 else ''})")
    print("  " + "─" * 76)
    for i, r in enumerate(classement[:top_k], 1):
        flag = " 🏢 Lore partagé " if r["entreprise_matchee"] else ""
        marqueur_indet = " ❓" if r.get("guards_statut") == "INDETERMINE" else ""
        print(
            f"  {i:>2}. {r['cv_id']:<14} "
            f"technos={r['score_technos']:.2f}  "
            f"({r['seniorite_totale']:.1f} ans"
            + (f"/{r['annees_requises']:.0f} requis" if r.get('annees_requises') else "")
            + ")  "
            f"{flag}{marqueur_indet}"
            f"  dispo={r['disponibilite']}"
            )
        ig = r.get("indicateur_global")
        if ig:
            print(f"     🌐 affinité offre : cos {ig['cosine']:.4f}  ·  rang {ig['rang']}/{ig['sur']}")
        # Compatibilité : details peut être {label: float} (ancien) ou {label: dict} (nouveau)
        def _score(v):
            return v["score"] if isinstance(v, dict) else v

        ICONE_SOURCE = {"exact": "🟢 ", "semantique": "🟠 ", "absent": "🔴 "}

        manquantes = [
            (t, d) for t, d in r["technos_details"].items()
            if _score(d) < 0.5
        ]
        presentes = [
            (t, d) for t, d in r["technos_details"].items()
            if _score(d) >= 0.5
        ]

        if presentes:
            lignes = []
            for t, d in presentes:
                if isinstance(d, dict):
                    icone = ICONE_SOURCE.get(d["source"], "?")
                    match = f"→{d['matche_avec']}" if d.get("matche_avec") and d["matche_avec"] != t else ""
                    lignes.append(f"{icone}{t}{match}={_score(d):.2f}")
                else:
                    lignes.append(f"🟢 {t}={d:.2f}")
            print(f"     technos OK    : {', '.join(lignes[:8])}"
                  + (" ..." if len(lignes) > 8 else ""))

        if manquantes:
            lignes = []
            for t, d in manquantes:
                if isinstance(d, dict):
                    icone = ICONE_SOURCE.get(d["source"], "?")
                    match = f"→{d['matche_avec']}" if d.get("matche_avec") and d["matche_avec"] != t else ""
                    lignes.append(f"{icone}{t}{match}={_score(d):.2f}")
                else:
                    lignes.append(f"🔴 {t}={d:.2f}")
            print(f"      technos manquantes : {', '.join(lignes[:5])}"
                  + (" ..." if len(lignes) > 5 else ""))
    if len(classement) > top_k:
        print(f"      ... et {len(classement) - top_k} autres CV(s)")
    print()


def _afficher_rejetes(cv_rejetes: List[Dict]) -> None:
    if not cv_rejetes:
        return

    print(f"🚫 CV ÉCARTÉS PAR LES GARDE-FOUS  ({len(cv_rejetes)})\n")
    print("  " + "─" * 76)
    for r in cv_rejetes:
        ig = r.get("indicateur_global") or {}
        cos_txt = f"cos={ig['cosine']:.3f}" if ig else "cos=?"
        print(f"  ✗ {r['cv_id']:<14} "
              f"{cos_txt}  "
              f"[{r['categorie']}]")
        print(f"      statut : {r['guards_statut']}")
        print(f"      motif  : {r['guards_motif']}")
        d = r.get("guards_details", {})
        if "exp_pertinente" in d:
            print(f"      exp pertinente : {d['exp_pertinente']} "
                  f"(cosine={d['cosine_exp']:.3f}, "
                  f"fin il y a {d['mois_depuis_fin']} mois)")
        print()


def afficher_resultats(
    resultats_acceptes_par_categorie: Dict[str, List[Dict]],
    cv_rejetes: List[Dict],
    offre_id: str,
    poste_ao: str,
    top_k: int = 10,
) -> None:
    print(f"\n{'═' * 78}")
    print(f"  AO : {offre_id}  —  Poste demandé : {poste_ao}")
    print(f"{'═' * 78}\n")

    principal = {
        nom: cl for nom, cl in resultats_acceptes_par_categorie.items()
        if cl and cl[0]["est_principal"]
    }
    alternatifs = {
        nom: cl for nom, cl in resultats_acceptes_par_categorie.items()
        if cl and not cl[0]["est_principal"]
    }

    if principal:
        print(f"🏆 GROUPE PRINCIPAL\n")
        for nom, cl in principal.items():
            _afficher_groupe(nom, cl, top_k=top_k)
    else:
        print(f"🏆 GROUPE PRINCIPAL  (aucun CV avec ces qualifications)\n")

    if alternatifs:
        print(f"📂 PROFILS ALTERNATIFS\n")
        for nom, cl in sorted(alternatifs.items(), key=lambda kv: -len(kv[1])):
            _afficher_groupe(nom, cl, top_k=top_k)

    _afficher_rejetes(cv_rejetes)


# ════════════════════════════════════════════════════════════════════
# POINTS D'ENTRÉE PROGRAMMATIQUES (pour le backend / front)
# Emballent la logique du __main__ sans la modifier.


_GRAPHE_CACHE = None

def _graphe_compile():
    """Compile le graphe une seule fois (réutilisé entre appels)."""
    global _GRAPHE_CACHE
    if _GRAPHE_CACHE is None:
        _GRAPHE_CACHE = construire_graphe()
    return _GRAPHE_CACHE


def lister_offres(dossier: str = "./AO_JSON") -> List[Dict]:
    """Offres prêtes (présentes dans AO_JSON), pour le sélecteur du front."""
    out = []
    for o in charger_offres(dossier):
        out.append({"id": o["id"], "poste": o["data"].get("poste", "?")})
    return out


def lister_cvs(dossier: str = "./CV_JSON") -> List[Dict]:
    """CV prêts (présents dans CV_JSON)."""
    return [{"id": c["id"]} for c in charger_cvs(dossier)]

def charger_seuils(chemin: str = "config.json") -> dict: # mis ici pour permettre de prendre en compte le changement utilisateur
    with open(chemin, "r", encoding="utf-8") as f:
        config = json.load(f)
    seuils = {
        "seuil_court_mois":  config["seuil_court_mois"],
        "seuil_valide_mois": config["seuil_valide_mois"],
    }
    return seuils

def lancer_matching(offre_id: str) -> Dict:
    """
    Équivalent programmatique du __main__ pour UNE offre choisie.
    Retourne un dict structuré (pas d'affichage console).
    """
    cvs    = charger_cvs("./CV_JSON")
    offres = charger_offres("./AO_JSON")
    offre  = next((o for o in offres if o["id"] == offre_id), None)
    if offre is None:
        raise ValueError(f"AO introuvable dans ./AO_JSON : {offre_id}")
    if not cvs:
        raise ValueError("Aucun CV dans ./CV_JSON")

    config_seuils = charger_seuils()   
    state_final = _graphe_compile().invoke(state_initial(offre, cvs, config_seuils))
    return {
        "offre_id":  offre["id"],
        "poste_ao":  offre["data"].get("poste", "?"),
        "resultats_acceptes_par_categorie":
            state_final.get("resultats_acceptes_par_categorie") or {},
        "cv_rejetes": state_final.get("cv_rejetes") or [],
        "erreurs":    state_final.get("erreurs") or [],
    }


# ════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE


if __name__ == "__main__":
    print("Chargement des données...")
    cvs    = charger_cvs("./CV_JSON")
    offres = charger_offres("./AO_JSON")
    print(f"   {len(cvs)} CVs et {len(offres)} offres chargés.\n")

    if not cvs or not offres:
        print("❌Aucune donnée à traiter.")
        exit(1)

    config_seuils = charger_seuils()

    print("Construction du graphe LangGraph...")
    graphe = construire_graphe()
    print("Graphe prêt.\n")

    offre_cible = offres[0]
    poste_ao    = offre_cible["data"].get("poste", "?")
    print(f"Matching pour l'offre {offre_cible['id']} ({poste_ao})...")

    state_init  = state_initial(offre_cible, cvs, config_seuils)
    state_final = graphe.invoke(state_init)

    if state_final["erreurs"]:
        print("\n⚠️  Erreurs rencontrées :")
        for err in state_final["erreurs"]:
            print(f"   - {err}")

    afficher_resultats(
        resultats_acceptes_par_categorie = state_final.get("resultats_acceptes_par_categorie") or {},
        cv_rejetes                       = state_final.get("cv_rejetes") or [],
        offre_id = offre_cible["id"],
        poste_ao = poste_ao,
    )