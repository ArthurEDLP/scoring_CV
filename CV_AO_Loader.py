import json
from pathlib import Path
from typing import List, Dict

def charger_cvs(dossier: str) -> List[Dict]:
    """
    Charge tous les CVs depuis un dossier contenant des fichiers JSON.
    1 fichier JSON = 1 CV.

    Args:
        dossier (str): chemin vers le dossier

    Returns:
        List[Dict]: liste de CVs structurés avec 'id', 'source', 'data'
    """
    tous_les_cvs = []

    for fichier in Path(dossier).glob("*.json"):
        with open(fichier, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            data = data[0]

        cv = {
            "id":     data.get("data", data).get("id", fichier.stem),
            "source": str(fichier),
            "data":   data.get("data", data)
        }

        tous_les_cvs.append(cv)

    return tous_les_cvs


def charger_offres(dossier: str) -> List[Dict]:
    """
    Charge toutes les offres depuis un dossier contenant des fichiers JSON.
    1 fichier JSON = 1 offre.

    Args:
        dossier (str): chemin vers le dossier

    Returns:
        List[Dict]: liste d'offres structurées avec 'id', 'source', 'data'
    """
    toutes_les_offres = []

    for fichier in Path(dossier).glob("*.json"):
        with open(fichier, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            data = data[0]

        offre = {
            "id":     data.get("data", data).get("id", fichier.stem),
            "source": str(fichier),
            "data":   data.get("data", data)
        }

        toutes_les_offres.append(offre)

    return toutes_les_offres