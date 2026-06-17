"""
Cache d'embeddings pour CV et AO.

Stocke par CV :
  - les expériences (poste embeddé + durée + entreprise + index pour ordre)
  - les technos (chacune embeddée)
  - la séniorité totale précalculée (somme de toutes les durées)

Stocke par AO :
  - le poste embeddé
  - les technos embeddées

Mécanique de cache : hash MD5 du contenu, recalcul auto si modifié.
"""

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

import ollama

from score_global import texte_experience


MODEL = "qwen3-embedding:8b"

# ─────────────────────────────────────────────────────────────────────
# Normalisation des vecteurs d'embedding
# ─────────────────────────────────────────────────────────────────────

def _normaliser(vecteurs: np.ndarray) -> np.ndarray:
    """Normalise L2 ligne par ligne. Accepte (n, d) ou (d,).
    Remplace les normes nulles par 1.0 pour éviter NaN."""
    arr = np.asarray(vecteurs, dtype="float32")
    if arr.ndim == 1:
        n = np.linalg.norm(arr)
        return arr if n == 0 else arr / n
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms

# ─────────────────────────────────────────────────────────────────────
# Fonction d'embeding générique
# ─────────────────────────────────────────────────────────────────────

def embed(textes):
    textes = [t[:8000] for t in textes]      # garde-fou longueur de contexte car j'ai des CVs très gros
    rep = ollama.embed(model=MODEL, input=textes, keep_alive=-1)
    return np.array(rep["embeddings"], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────
# Templates d'enrichissement
# ─────────────────────────────────────────────────────────────────────

def _template_techno(techno: str) -> str:
    return f"compétence technique en {techno.strip()}"

def _template_poste(poste: str) -> str:
    return f"poste de {poste.strip()}"


# ─────────────────────────────────────────────────────────────────────
# Sérialisation numpy <-> JSON
# ─────────────────────────────────────────────────────────────────────

def _ndarray_to_json(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
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

_RE_DATE = re.compile(r"(\d{1,2})/(\d{4})")

# Mots indiquant "expérience en cours" (à la place d'une 2ème date)
_RE_EN_COURS = re.compile(
    r"(aujourd['’]?hui|present|présent|current|ongoing|en cours|nowadays|now)",
    re.IGNORECASE,
)

# Plafond pour les expériences avec une seule date ET pas de mention "en cours".
DUREE_DEFAUT_DATE_UNIQUE = 0.25   # 3 mois (typiquement un stage court)

# Plafond pour les expériences "stage/stagiaire/intern".
DUREE_MAX_STAGE = 0.5             # 6 mois max
_RE_STAGE = re.compile(r"\b(stage|stagiaire|intern|internship)\b", re.IGNORECASE)


def _est_stage(intitule_poste: str) -> bool:
    return bool(_RE_STAGE.search(intitule_poste or ""))


def _duree_experience_annees(date_str: str, poste: str = "") -> float:
    """
    Calcule la durée en années depuis une chaîne 'MM/YYYY - MM/YYYY'
    ou 'MM/YYYY - aujourd'hui'.

    Règles :
      - 2 dates explicites → durée réelle
      - 1 date + mot "en cours" (aujourd'hui, present...) → durée jusqu'à
        maintenant
      - 1 date sans mention → DUREE_DEFAUT_DATE_UNIQUE (3 mois, anti-bug)
      - Stages plafonnés à 6 mois
      - Format invalide → 0.0
    """
    if not date_str:
        return 0.0

    matches = _RE_DATE.findall(date_str)
    if not matches:
        return 0.0

    try:
        m1, y1 = matches[0]
        debut = datetime(int(y1), int(m1), 1)
    except (ValueError, IndexError):
        return 0.0

    if len(matches) >= 2:
        # Deux dates explicites → durée réelle
        try:
            m2, y2 = matches[1]
            fin = datetime(int(y2), int(m2), 1)
        except ValueError:
            fin = datetime.now()
        delta_jours = (fin - debut).days
        duree = max(0.0, delta_jours / 365.25)
    elif _RE_EN_COURS.search(date_str):
        # Une date + mot "en cours" → durée jusqu'à maintenant
        delta_jours = (datetime.now() - debut).days
        duree = max(0.0, delta_jours / 365.25)
    else:
        # Une seule date sans mention → courte durée par défaut
        duree = DUREE_DEFAUT_DATE_UNIQUE

    # Plafond pour les stages, quoi qu'il arrive
    if _est_stage(poste):
        duree = min(duree, DUREE_MAX_STAGE)

    return duree


# ─────────────────────────────────────────────────────────────────────
# Cache CV
# ─────────────────────────────────────────────────────────────────────

class CacheEmbeddingsCV:
    """
    Pour chaque CV, on cache :
      - une liste d'expériences (poste embeddé + durée + entreprise + ordre)
      - une liste de technos (chacune embeddée séparément)
      - la séniorité totale (somme de toutes les durées)
    """

    def __init__(self, model: MODEL, cache_dir: str):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ───── I/O fichier ─────

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
        """Embedde les postes d'expérience et chaque techno."""

        # 1. Expériences : on conserve l'ordre d'origine (le 1er = le plus récent
        #    par convention dans les CVs)
        experiences = []
        for idx, exp in enumerate(cv_data.get("experiences", [])):
            poste = exp.get("poste", "").strip()
            if not poste:
                continue
            entreprise = exp.get("entreprise", "").strip()
            duree = _duree_experience_annees(exp.get("date", ""), poste)

            texte = texte_experience(exp)               # poste + entreprise + détails
            emb_raw = embed([texte])[0]

            emb = _normaliser(emb_raw)

            experiences.append({
                "ordre":      idx,                   # 0 = la plus récente
                "poste":      poste,
                "annees":     round(duree, 3),
                "entreprise": entreprise,
                "embedding":  _ndarray_to_json(emb),
            })

        # 2. Séniorité totale = somme de TOUTES les expériences
        seniorite_totale = round(sum(e["annees"] for e in experiences), 2)

        # 3. Technos : chaque techno embeddée séparément
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

        return {
            "experiences":      experiences,
            "technos":          technos,
            "seniorite_totale": seniorite_totale,
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
            "seniorite_totale": <float>
          }
        """
        cv_id   = cv["id"]
        data    = cv["data"]
        hash_ok = _hash_contenu(data)
        cache   = self._lire(cv_id)

        if cache and cache.get("hash") == hash_ok:
            vecteurs_json = cache["vecteurs"]
        else:
            vecteurs_json = self._calculer(data)
            self._ecrire(cv_id, hash_ok, vecteurs_json)

        return self._desserialiser(vecteurs_json)

    def _desserialiser(self, vecteurs_json: dict) -> Dict:
        return {
            "experiences": [
                {
                    "ordre":      e.get("ordre", i),
                    "poste":      e["poste"],
                    "annees":     e["annees"],
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
        }

    def invalider(self, cv_id: str):
        chemin = self._chemin(cv_id)
        if chemin.exists():
            chemin.unlink()


# ─────────────────────────────────────────────────────────────────────
# Cache AO
# ─────────────────────────────────────────────────────────────────────

class CacheEmbeddingsOffre:
    """Cache des embeddings AO : poste + technos."""

    def __init__(self, model: MODEL, cache_dir: str):
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

        return {"poste": poste_obj, "technos": technos}

    def obtenir(self, offre: dict) -> Dict:
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
        }

    def invalider(self, offre_id: str):
        chemin = self._chemin(offre_id)
        if chemin.exists():
            chemin.unlink()
