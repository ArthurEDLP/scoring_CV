"""
Taxonomie des technos : familles fonctionnelles + gate de compatibilité.

Sert au filtrage des matchs sémantiques dans score_technos : on bloque les
rapprochements entre familles incompatibles (Airflow↔Slack, dbt↔Scala...),
quel que soit le cosinus.

Usage côté scoring.py :
    from taxonomie import compatibles_technos
    ...
    if not compatibles_technos(label_ao, label_cv):
        continue   # couple écarté avant même le cosinus

Le fichier de données (familles_technos.json) est cherché à côté de ce module,
indépendamment du répertoire de travail.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

# Chemin par défaut : le JSON est à côté de ce fichier .py
_CHEMIN_DEFAUT = Path(__file__).with_name("familles_technos.json")

# Caches chargés une seule fois (lazy : au premier appel, pas à l'import)
_FAMILLES: Optional[dict] = None
_COMPATIBLES: Optional[set] = None

# Pour ta curation : technos rencontrées mais absentes de la table
TECHNOS_NON_CLASSEES: set[str] = set()


def charger_familles(chemin: Path | str = _CHEMIN_DEFAUT) -> None:
    """Charge (ou recharge) la table depuis le JSON. Idempotent."""
    global _FAMILLES, _COMPATIBLES
    data = json.loads(Path(chemin).read_text(encoding="utf-8"))
    _FAMILLES = {k.strip().lower(): v for k, v in data["familles"].items()}
    _COMPATIBLES = {frozenset(p) for p in data["compatibles"]}


def _assurer_charge() -> None:
    if _FAMILLES is None:
        charger_familles()


def famille(label: str) -> Optional[str]:
    """Famille d'une techno, ou None si non classée (et on la logue)."""
    _assurer_charge()
    fam = _FAMILLES.get(label.strip().lower())
    if fam is None:
        TECHNOS_NON_CLASSEES.add(label.strip().lower())
    return fam


def compatibles_technos(a: str, b: str) -> bool:
    """
    True si un match sémantique entre a et b est autorisé.
    - famille inconnue d'un côté -> True (fallback permissif : cosinus seul)
    - même famille -> True
    - familles différentes -> True seulement si la paire est dans 'compatibles'
    """
    _assurer_charge()
    fa, fb = famille(a), famille(b)
    if fa is None or fb is None:
        return True
    if fa == fb:
        return True
    return frozenset({fa, fb}) in _COMPATIBLES
