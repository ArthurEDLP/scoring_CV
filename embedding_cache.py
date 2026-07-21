"""
Cache d'embeddings pour CV et AO.

Stocke par CV :
  - les expériences (poste embeddé + durée + entreprise + index pour ordre)
  - les technos (chacune embeddée)
  - la séniorité totale précalculée (somme de toutes les durées)
  - le CV complet embeddé (concaténation expériences + technos + soft skills)

Stocke par AO :
  - le poste embeddé
  - les technos embeddées
  - l'AO complète embeddée (concaténation Profil + Description + Contexte)

Mécanique de cache : hash MD5 du contenu, recalcul auto si modifié.

Les embeddings "complets" (ao_complete / cv_complet) sont là pour les fonctionnalités d'affichage et de matching global (ex: top_experiences),
sans devoir rappeler Ollama à chaque matching.
"""

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from score_global import texte_experience, texte_ao_complet, _texte_section_ao, texte_cv_complet, _normaliser, embed, MODEL


# ─────────────────────────────────────────────────────────────────────
# Templates d'enrichissement
# ─────────────────────────────────────────────────────────────────────

# l'utilité est de donnée un contexte sémantique au modèle. Ce n'est plus un simple mot ou groupement de mots, c'est une compétence technique ou un poste

def _template_techno(techno: str) -> str:
    return f"compétence technique en {techno.strip()}"

def _template_poste(poste: str) -> str:
    return f"poste de {poste.strip()}"


# ─────────────────────────────────────────────────────────────────────
# Sérialisation numpy <-> JSON
# ─────────────────────────────────────────────────────────────────────

# un fichier JSON ne stocke pas du NumPy, l'embeddding est ici du NumPy. Il faut la transformation

def _ndarray_to_json(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape), # liste au lieu de tuple
        "dtype": str(arr.dtype),
        "data":  arr.tolist(),
    }

def _json_to_ndarray(obj: dict) -> np.ndarray:
    return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])


# ─────────────────────────────────────────────────────────────────────
# Hash de contenu
# ─────────────────────────────────────────────────────────────────────

def _hash_contenu(data: dict) -> str:
    contenu = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(contenu.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Parsing des dates d'expérience
# ─────────────────────────────────────────────────────────────────────

# Mois en toutes lettres (FR + abréviations courantes) -> numéro
_MOIS = {
    "janvier": 1, "janv": 1, "jan": 1,
    "fevrier": 2, "février": 2, "fevr": 2, "févr": 2, "fev": 2, "fév": 2,
    "mars": 3, "mar": 3,
    "avril": 4, "avr": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7, "juil": 7, "jui": 7,
    "aout": 8, "août": 8,
    "septembre": 9, "sept": 9, "sep": 9,
    "octobre": 10, "oct": 10,
    "novembre": 11, "nov": 11,
    "decembre": 12, "décembre": 12, "dec": 12, "déc": 12,
}

# "Mars 2022", "Septembre 2019", "Févr. 2022"
_RE_MOIS_TXT = re.compile(
    r"\b(" + "|".join(sorted(_MOIS, key=len, reverse=True)) + r")\b\.?\s+(\d{4})",
    re.IGNORECASE,
)
# "MM/YYYY", "MM.YYYY", "MM-YYYY"
_RE_MOIS_NUM = re.compile(r"\b(\d{1,2})[./\-](\d{4})\b")
# Année seule (dernier recours : "2018")
_RE_ANNEE = re.compile(r"\b(\d{4})\b")

# Mots/préfixes indiquant une expérience EN COURS
_RE_EN_COURS = re.compile(
    r"(aujourd['’]?hui|presents?|présents?|current|ongoing|en cours|"
    r"nowadays|now|depuis)",
    re.IGNORECASE,
)

# Plafond pour les expériences avec une seule date ET pas de mention "en cours".
DUREE_DEFAUT_DATE_UNIQUE = 0.25   # 3 mois (un stage court)

# Plafond pour les expériences "stage/stagiaire/intern".
_RE_STAGE_ALTERNANCE = re.compile(
    r"\b(stage|stagiaire|alternance|alternant|apprentissage|intern|internship)\b",
    re.IGNORECASE)
# Tous les tirets "longs"/exotiques -> trait d'union normal "-"
_TIRETS = dict.fromkeys(map(ord, "–—‒―−﹘﹣－"), "-")

def _normaliser_tirets(s: str) -> str:
    return s.translate(_TIRETS)

def _est_stage_alternance(intitule_poste: str) -> bool:
    return bool(_RE_STAGE_ALTERNANCE.search(intitule_poste or ""))


def _extraire_dates(date_str: str):
    """
    Retourne la liste des (mois, annee) trouvés, dans l'ordre d'apparition.
    Reconnaît : mois en lettres ('Mars 2022'), MM/YYYY, MM.YYYY, MM-YYYY,
    et en dernier recours l'année seule ('2018' -> janvier).
    """
    date_str = _normaliser_tirets(date_str)
    trouve = []  # (position, mois, annee)
    for m in _RE_MOIS_TXT.finditer(date_str):
        cle = m.group(1).lower().rstrip(".")
        trouve.append((m.start(), _MOIS[cle], int(m.group(2))))
    for m in _RE_MOIS_NUM.finditer(date_str):
        mois, an = int(m.group(1)), int(m.group(2))
        if 1 <= mois <= 12:
            trouve.append((m.start(), mois, an))
    if not trouve:  # rien de précis -> on tente l'année seule
        for m in _RE_ANNEE.finditer(date_str):
            trouve.append((m.start(), 1, int(m.group(1))))
    trouve.sort()
    return [(mo, an) for _, mo, an in trouve]


def _duree_experience_annees(date_str: str, poste: str = "") -> float:
    """
    Durée en années à partir d'une chaîne de période. Gère :
      - 2 dates explicites           -> durée réelle
      - 1 date + mention "en cours"  -> durée jusqu'à maintenant
      - 1 date sans mention          -> DUREE_DEFAUT_DATE_UNIQUE (3 mois)
      - stages et alternance         -> Ce n'est pas comptabilisé 
      - format invalide              -> 0.0
    Formats reconnus : 'Mars 2022', '07.2022', '03/2020', '2018', etc.
    """
    if not date_str:
        return 0.0

    dates = _extraire_dates(date_str)
    if not dates:
        return 0.0

    try:
        m1, y1 = dates[0]
        debut = datetime(y1, m1, 1)
    except (ValueError, IndexError):
        return 0.0

    if len(dates) >= 2:
        try:
            m2, y2 = dates[1]
            fin = datetime(y2, m2, 1)
        except ValueError:
            fin = datetime.now()
        duree = max(0.0, (fin - debut).days / 365.25)
    elif _RE_EN_COURS.search(date_str):
        duree = max(0.0, (datetime.now() - debut).days / 365.25)
    else:
        duree = DUREE_DEFAUT_DATE_UNIQUE

    if _est_stage_alternance(poste):
        return 0.0

    return duree


def _anciennete_experience_annees(date_str: str): # problème ici, car une fois fais, la date "aujourd'hui" est fixe, sauf si on embed tout les jours
    """
    Années écoulées depuis la FIN de l'expérience.
      - 0.0 si l'expérience est en cours ('présent', 'aujourd'hui', 'depuis'…)
      - années entre la date de fin et aujourd'hui sinon
      - None si aucune date exploitable (le CV sera exclu du principal)
    """
    if not date_str:
        return None
    if _RE_EN_COURS.search(_normaliser_tirets(date_str)):
        return 0.0
    dates = _extraire_dates(date_str)
    if not dates:
        return None
    m, y = dates[-1]                      # date de fin = la dernière trouvée
    try:
        fin = datetime(y, m, 1)
    except ValueError:
        return None
    return max(0.0, (datetime.now() - fin).days / 365.25)


# ─────────────────────────────────────────────────────────────────────
# Cache CV
# ─────────────────────────────────────────────────────────────────────

class CacheEmbeddingsCV:
    """
    Pour chaque CV, on cache :
      - une liste d'expériences (poste embeddé + durée + entreprise + ordre)
      - une liste de technos (chacune embeddée séparément)
      - la séniorité totale (somme de toutes les durées)
      - l'embedding du CV complet (expériences + technos + savoir-faire/être)
    """

    def __init__(self, model: str, cache_dir: str):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _chemin(self, cv_id: str) -> Path:
        safe_id = cv_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_id}.json"

    def _lire(self, cv_id: str) -> Optional[dict]:
        chemin = self._chemin(cv_id)
        if not chemin.exists():
            return None
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ecrire(self, cv_id: str, hash_contenu: str, vecteurs_json: dict):
        payload = {
            "id":       cv_id,
            "hash":     hash_contenu,
            "vecteurs": vecteurs_json,
        }
        with open(self._chemin(cv_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ───── Calcul ─────

    def _calculer(self, cv_data: dict) -> dict:
        """Embedde les postes d'expérience, chaque techno, et le CV complet."""

        # Expériences : on conserve l'ordre d'origine (le 1er = le plus récent par convention dans les CVs)
        experiences = []
        for idx, exp in enumerate(cv_data.get("experiences", [])): # boucle des expériences d'un CV
            poste = exp.get("poste", "").strip()
            if not poste:
                continue
            entreprise = exp.get("entreprise", "").strip()
            duree = _duree_experience_annees(exp.get("date", ""), poste) # le poste est important car il permet de différencier d'un stage6
            anciennete = _anciennete_experience_annees(exp.get("date", ""))

            texte = texte_experience(exp)   # poste + entreprise + détails
            emb_raw = embed([texte])[0]
            emb = _normaliser(emb_raw)

            experiences.append({
                "ordre":      idx,  # 0 = la plus récente
                "poste":      poste,
                "annees":     round(duree, 3),
                "anciennete": (round(anciennete, 3) if anciennete is not None else None),
                "entreprise": entreprise,
                "embedding":  _ndarray_to_json(emb),
            })

        # Séniorité totale = somme de TOUTES les expériences
        seniorite_totale = round(sum(e["annees"] for e in experiences), 2)

        # Technos : chaque techno embeddée séparément
        technos_brutes = cv_data.get("competences_techniques", [])
        technos_brutes = [t.strip() for t in technos_brutes if t and t.strip()]

        technos = []
        if technos_brutes:
            embs_raw = embed([_template_techno(t) for t in technos_brutes])
            embs = _normaliser(embs_raw)

            for label, emb in zip(technos_brutes, embs):
                technos.append({
                    "label":     label,
                    "embedding": _ndarray_to_json(emb),
                })

        # CV complet (toutes les sections textuelles concaténées)
        texte_complet = texte_cv_complet(cv_data)
        if texte_complet:
            emb_raw = embed([texte_complet])[0]
            emb_complet = _normaliser(emb_raw)
            cv_complet_obj = {
                "texte":     texte_complet[:200] + ("..." if len(texte_complet) > 200 else ""),
                "embedding": _ndarray_to_json(emb_complet),
            }
        else:
            cv_complet_obj = None

        return {
            "experiences":      experiences,
            "technos":          technos,
            "seniorite_totale": seniorite_totale,
            "cv_complet":       cv_complet_obj,
        }

    # ───── API publique ─────

    def obtenir(self, cv: dict) -> Dict:
        """
        Retourne pour un CV :
          {
            "experiences": [
              {"ordre":0, "poste":..., "annees":..., "entreprise":..., "embedding": np.ndarray},
              ...
            ],
            "technos": [
              {"label":..., "embedding": np.ndarray}, ...
            ],
            "seniorite_totale": <float>,
            "cv_complet": {"texte": <preview>, "embedding": np.ndarray} | None,
          }
        """
        cv_id   = cv["id"]
        data    = cv["data"]
        hash_ok = _hash_contenu(data)
        cache   = self._lire(cv_id)

        if cache and cache.get("hash") == hash_ok:
            vecteurs_json = cache["vecteurs"]
        else:                                       # si hash différent on calcule les embeddings
            vecteurs_json = self._calculer(data)
            self._ecrire(cv_id, hash_ok, vecteurs_json)

        return self._desserialiser(vecteurs_json)

    def _desserialiser(self, vecteurs_json: dict) -> Dict:
        """
        Retransforme les données stockées, en JSON, en objets NumPy
        """
        cv_complet_obj = vecteurs_json.get("cv_complet")
        return {
            "experiences": [
                {
                    "ordre":      e.get("ordre", i),
                    "poste":      e["poste"],
                    "annees":     e["annees"],
                    "anciennete": e.get("anciennete"),
                    "entreprise": e["entreprise"],
                    "embedding":  _json_to_ndarray(e["embedding"]),
                }
                for i, e in enumerate(vecteurs_json.get("experiences", []))
            ],
            "technos": [
                {
                    "label":     t["label"],
                    "embedding": _json_to_ndarray(t["embedding"]),
                }
                for t in vecteurs_json.get("technos", [])
            ],
            "seniorite_totale": vecteurs_json.get("seniorite_totale", 0.0),
            "cv_complet": (
                {
                    "texte":     cv_complet_obj.get("texte", ""),
                    "embedding": _json_to_ndarray(cv_complet_obj["embedding"]),
                }
                if cv_complet_obj else None
            ),
        }



# ─────────────────────────────────────────────────────────────────────
# Cache AO
# ─────────────────────────────────────────────────────────────────────

class CacheEmbeddingsOffre:
    """
    Cache des embeddings AO :
      - poste
      - technos
      - AO complète (Profil + Description + Contexte)
    """

    def __init__(self, model: str, cache_dir: str):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _chemin(self, offre_id: str) -> Path:
        safe_id = offre_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_id}.json"

    def _lire(self, offre_id: str) -> Optional[dict]:
        chemin = self._chemin(offre_id)
        if not chemin.exists():
            return None
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ecrire(self, offre_id: str, hash_contenu: str, vecteurs_json: dict):
        payload = {
            "id":       offre_id,
            "hash":     hash_contenu,
            "vecteurs": vecteurs_json,
        }
        with open(self._chemin(offre_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _calculer(self, offre_data: dict) -> dict:
        # Poste
        poste = (offre_data.get("poste") or "").strip()
        if poste:
            emb_raw = embed([_template_poste(poste)])[0]
            emb_poste = _normaliser(emb_raw)

            poste_obj = {
                "label":     poste,
                "embedding": _ndarray_to_json(emb_poste),
            }
        else:
            poste_obj = None

        # Technos
        technos_brutes = offre_data.get("technos", [])
        technos_brutes = [t.strip() for t in technos_brutes if t and t.strip()]

        technos = []
        if technos_brutes:
            embs_raw = embed([_template_techno(t) for t in technos_brutes])
            embs = _normaliser(embs_raw)

            for label, emb in zip(technos_brutes, embs):
                technos.append({
                    "label":     label,
                    "embedding": _ndarray_to_json(emb),
                })

        # AO entier (Profil + Description + Contexte concaténés)
        texte_complet = texte_ao_complet(offre_data)
        if texte_complet:
            emb_raw = embed([texte_complet])[0]
            emb_complet = _normaliser(emb_raw)
            ao_complete_obj = {
                "texte":     texte_complet[:200] + ("..." if len(texte_complet) > 200 else ""),
                "embedding": _ndarray_to_json(emb_complet),
            }
        else:
            ao_complete_obj = None

        # Sections isolées (Profil / Description / Contexte) pour score_pertinence_cv
        sections = {}
        for cle, nom in (("profil", "Profil"),
                         ("description", "Description"),
                         ("contexte", "Contexte")):
            txt = _texte_section_ao(offre_data, nom)
            if txt:
                emb_section = _normaliser(embed([txt])[0])
                sections[cle] = {"embedding": _ndarray_to_json(emb_section)}
            else:
                sections[cle] = None

        return {
            "poste":       poste_obj,
            "technos":     technos,
            "ao_complete": ao_complete_obj,
            "sections":    sections,
        }

    def obtenir(self, offre: dict) -> Dict:
        """
        Retourne pour une AO :
          {
            "poste":   {"label":..., "embedding": np.ndarray} | None,
            "technos": [{"label":..., "embedding": np.ndarray}, ...],
            "ao_complete": {"texte": <preview>, "embedding": np.ndarray} | None,
          }
        """
        offre_id = offre["id"]
        data     = offre["data"]
        hash_ok  = _hash_contenu(data)
        cache    = self._lire(offre_id)

        if cache and cache.get("hash") == hash_ok:
            vecteurs_json = cache["vecteurs"]
        else:
            vecteurs_json = self._calculer(data)
            self._ecrire(offre_id, hash_ok, vecteurs_json)

        return self._desserialiser(vecteurs_json)

    def _desserialiser(self, vecteurs_json: dict) -> Dict:
        poste_obj = vecteurs_json.get("poste")
        ao_complete_obj = vecteurs_json.get("ao_complete")
        return {
            "poste": (
                {
                    "label":     poste_obj["label"],
                    "embedding": _json_to_ndarray(poste_obj["embedding"]),
                }
                if poste_obj else None
            ),
            "technos": [
                {
                    "label":     t["label"],
                    "embedding": _json_to_ndarray(t["embedding"]),
                }
                for t in vecteurs_json.get("technos", [])
            ],
            "ao_complete": (
                {
                    "texte":     ao_complete_obj.get("texte", ""),
                    "embedding": _json_to_ndarray(ao_complete_obj["embedding"]),
                }
                if ao_complete_obj else None
            ),
            "sections": {
                cle: (_json_to_ndarray(obj["embedding"]) if obj else None)
                for cle, obj in (vecteurs_json.get("sections") or {}).items()
            },
        }