"""
Comparaison des fichiers de benchmark d'embedding.

Prend un dossier contenant des fichiers benchmark_<modele>_<aoId>.json
et produit des tableaux comparatifs lisibles (CSV + Markdown).

Pour chaque AO trouvée dans les fichiers, génère 2 tableaux :
  1. cv_complet_vs_<section>   : score CV complet par modèle
  2. experiences_vs_<section>  : score MAX par expérience (le meilleur match)

Usage :
    python comparer_benchmarks.py
        → lit ./benchmark, écrit ./benchmark/comparaisons/

    python comparer_benchmarks.py --input ./benchmark --output ./compa
        → dossiers custom

    python comparer_benchmarks.py --ao PMU
        → ne traite que les benchmarks de l'AO 'PMU'
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Chargement des fichiers de benchmark
# ─────────────────────────────────────────────────────────────────────

def _charger_benchmarks(dossier_in: Path,
                        ao_filtre: Optional[str] = None) -> List[Dict]:
    """
    Charge tous les benchmark_*.json du dossier.
    Filtre éventuellement sur un id d'AO précis.
    """
    fichiers = list(dossier_in.glob("benchmark_*.json"))
    if not fichiers:
        raise RuntimeError(f"Aucun fichier benchmark_*.json dans {dossier_in}")

    benchmarks = []
    for f in fichiers:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            # Validation rapide
            if not all(k in data for k in ("modele", "ao", "resultats")):
                print(f"  ⚠️  {f.name} : structure invalide, ignoré")
                continue
            if ao_filtre and data["ao"].get("id") != ao_filtre:
                continue
            benchmarks.append({"fichier": f.name, **data})
        except Exception as e:
            print(f"  ⚠️  {f.name} : {e}")

    return benchmarks


# ─────────────────────────────────────────────────────────────────────
# Construction des tableaux
# ─────────────────────────────────────────────────────────────────────

SECTIONS = ["vs_profil", "vs_description", "vs_contexte", "vs_ao_complete"]


def _construire_tableau_cv_complet(
    benchmarks_d_une_ao: List[Dict],
    section: str,
) -> Tuple[List[str], List[List]]:
    """
    Tableau : pour chaque CV, score 'CV complet vs section' par modèle.

    Returns:
      (header, lignes)
      header : ["CV", "modele1", "modele2", ...]
      lignes : [["CV_JL", 0.74, 0.78, ...], ...]
    """
    modeles = [b["modele"] for b in benchmarks_d_une_ao]
    header = ["CV"] + modeles

    # cv_id -> {modele -> score}
    par_cv: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    for b in benchmarks_d_une_ao:
        for cv_id, contenu in b["resultats"].items():
            scores = contenu.get("scores", {})
            cv_complet = scores.get("cv_complet_vs_sections", {})
            par_cv[cv_id][b["modele"]] = cv_complet.get(section)

    lignes = []
    for cv_id in sorted(par_cv.keys()):
        ligne = [cv_id]
        for m in modeles:
            v = par_cv[cv_id].get(m)
            ligne.append(round(v, 4) if v is not None else None)
        lignes.append(ligne)

    return header, lignes


def _construire_tableau_experiences_max(
    benchmarks_d_une_ao: List[Dict],
    section: str,
) -> Tuple[List[str], List[List]]:
    """
    Tableau : pour chaque CV, score MAX parmi ses expériences vs section, par modèle.

    Le max est plus parlant que la moyenne : il dit 'cette personne a au moins une
    expérience pertinente'.
    """
    modeles = [b["modele"] for b in benchmarks_d_une_ao]
    header = ["CV"] + modeles

    par_cv: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    for b in benchmarks_d_une_ao:
        for cv_id, contenu in b["resultats"].items():
            scores = contenu.get("scores", {})
            exps = scores.get("experiences_vs_sections", {}).get(section, [])
            if exps:
                scores_exp = [e["score"] for e in exps if e.get("score") is not None]
                par_cv[cv_id][b["modele"]] = max(scores_exp) if scores_exp else None
            else:
                par_cv[cv_id][b["modele"]] = None

    lignes = []
    for cv_id in sorted(par_cv.keys()):
        ligne = [cv_id]
        for m in modeles:
            v = par_cv[cv_id].get(m)
            ligne.append(round(v, 4) if v is not None else None)
        lignes.append(ligne)

    return header, lignes


def _construire_tableau_cv_vs_exp(
    benchmarks_d_une_ao: List[Dict],
    section: str,
) -> Tuple[List[str], List[List]]:
    """
    Pour CHAQUE modèle, met côte à côte sur la même section :
      - le score 'CV complet'
      - le score 'meilleure expérience'
      - l'écart (exp_max - cv_complet)

    Permet de voir si prendre la meilleure expérience change significativement
    le résultat par rapport au CV complet concaténé.

    Header :
      ["CV", "<m1>_cvComplet", "<m1>_expMax", "<m1>_ecart", "<m2>_cvComplet", ...]

    L'écart est positif si la meilleure expérience score PLUS HAUT que le CV
    complet (→ le CV complet "dilue" le signal du bon poste).
    L'écart est négatif si le CV complet score plus haut (→ le contexte global
    aide plus qu'une expérience isolée).
    """
    modeles = [b["modele"] for b in benchmarks_d_une_ao]

    header = ["CV"]
    for m in modeles:
        header += [f"{m} · cv_complet", f"{m} · exp_max", f"{m} · ecart"]

    # cv_id -> modele -> (cv_complet, exp_max)
    donnees: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = \
        defaultdict(dict)

    for b in benchmarks_d_une_ao:
        for cv_id, contenu in b["resultats"].items():
            scores = contenu.get("scores", {})

            cv_complet = scores.get("cv_complet_vs_sections", {}).get(section)

            exps = scores.get("experiences_vs_sections", {}).get(section, [])
            scores_exp = [e["score"] for e in exps if e.get("score") is not None]
            exp_max = max(scores_exp) if scores_exp else None

            donnees[cv_id][b["modele"]] = (cv_complet, exp_max)

    lignes = []
    for cv_id in sorted(donnees.keys()):
        ligne: List = [cv_id]
        for m in modeles:
            cv_c, exp_m = donnees[cv_id].get(m, (None, None))
            ecart = (
                round(exp_m - cv_c, 4)
                if (cv_c is not None and exp_m is not None) else None
            )
            ligne += [
                round(cv_c, 4) if cv_c is not None else None,
                round(exp_m, 4) if exp_m is not None else None,
                ecart,
            ]
        lignes.append(ligne)

    return header, lignes


# ─────────────────────────────────────────────────────────────────────
# Écriture CSV + Markdown
# ─────────────────────────────────────────────────────────────────────

def _ecrire_csv(chemin: Path, header: List[str], lignes: List[List]):
    with open(chemin, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        for ligne in lignes:
            w.writerow([
                "" if v is None else
                (str(v).replace(".", ",") if isinstance(v, float) else v)
                for v in ligne
            ])


def _ecrire_markdown(chemin: Path, titre: str,
                     header: List[str], lignes: List[List]):
    """Tableau Markdown avec largeur de colonnes auto.

    Les '|' présents dans les cellules sont échappés en '\\|' pour ne pas
    casser la structure du tableau Markdown.
    """
    def _cell(v) -> str:
        if v is None:
            return ""
        s = f"{v:.4f}" if isinstance(v, float) else str(v)
        return s.replace("|", "\\|")

    rows = [[_cell(c) for c in header]] + [
        [_cell(v) for v in ligne] for ligne in lignes
    ]
    largeurs = [max(len(r[i]) for r in rows) for i in range(len(header))]

    def _format_row(r):
        return "| " + " | ".join(s.ljust(w) for s, w in zip(r, largeurs)) + " |"

    with open(chemin, "w", encoding="utf-8") as f:
        f.write(f"# {titre}\n\n")
        f.write(_format_row(rows[0]) + "\n")
        f.write("| " + " | ".join("-" * w for w in largeurs) + " |\n")
        for r in rows[1:]:
            f.write(_format_row(r) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Analyse simple (qui gagne sur quoi)
# ─────────────────────────────────────────────────────────────────────

def _resume_par_modele(benchmarks_d_une_ao: List[Dict]) -> str:
    """
    Calcule pour chaque modèle :
      - moyenne des scores 'CV complet vs AO complète'
      - écart-type (mesure de discrimination : plus c'est haut, plus le modèle
        sépare bien les bons des mauvais CVs)
    """
    lignes = []
    for b in benchmarks_d_une_ao:
        scores = []
        for cv_id, contenu in b["resultats"].items():
            s = contenu.get("scores", {}).get("cv_complet_vs_sections", {}).get("vs_ao_complete")
            if s is not None:
                scores.append(s)
        if not scores:
            continue
        moyenne = sum(scores) / len(scores)
        var = sum((s - moyenne) ** 2 for s in scores) / len(scores)
        ecart_type = var ** 0.5
        lignes.append({
            "modele": b["modele"],
            "nb_cv": len(scores),
            "moyenne": round(moyenne, 4),
            "ecart_type": round(ecart_type, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        })

    # Tri par écart-type décroissant (plus discriminant en premier)
    lignes.sort(key=lambda d: -d["ecart_type"])

    out = "## Résumé par modèle (sur 'CV complet vs AO complète')\n\n"
    out += "Tri par écart-type décroissant : plus c'est haut, plus le modèle "
    out += "discrimine entre bons et mauvais CVs.\n\n"
    out += "| Modèle | Nb CV | Moyenne | Écart-type | Min | Max |\n"
    out += "|---|---|---|---|---|---|\n"
    for l in lignes:
        out += (f"| {l['modele']} | {l['nb_cv']} | "
                f"{l['moyenne']} | **{l['ecart_type']}** | "
                f"{l['min']} | {l['max']} |\n")
    return out


def _resume_cv_vs_exp(benchmarks_d_une_ao: List[Dict]) -> str:
    """
    Pour chaque modèle et chaque section, calcule l'écart moyen
    (exp_max - cv_complet) sur tous les CVs.

    Interprétation :
      - écart moyen POSITIF  → la meilleure expérience score plus haut que
        le CV complet : prendre l'expérience est plus discriminant
        (le CV complet "noie" le signal).
      - écart moyen NÉGATIF  → le CV complet score plus haut : le contexte
        global apporte plus qu'une expérience isolée.
      - écart proche de 0    → peu importe la stratégie choisie.
    """
    out = "## CV complet vs Meilleure expérience\n\n"
    out += "Écart moyen = moyenne(exp_max - cv_complet) sur tous les CVs.\n\n"
    out += "- **Positif** : la meilleure expérience est plus discriminante "
    out += "(le CV complet dilue le signal)\n"
    out += "- **Négatif** : le CV complet est plus pertinent "
    out += "(le contexte global aide)\n"
    out += "- **~0** : les deux stratégies sont équivalentes\n\n"

    out += "| Modèle | Section | Écart moyen | exp_max (moy) | cv_complet (moy) |\n"
    out += "|---|---|---|---|---|\n"

    for b in sorted(benchmarks_d_une_ao, key=lambda x: x["modele"]):
        for section in SECTIONS:
            ecarts = []
            exp_vals = []
            cv_vals = []
            for cv_id, contenu in b["resultats"].items():
                scores = contenu.get("scores", {})
                cv_c = scores.get("cv_complet_vs_sections", {}).get(section)
                exps = scores.get("experiences_vs_sections", {}).get(section, [])
                se = [e["score"] for e in exps if e.get("score") is not None]
                exp_m = max(se) if se else None
                if cv_c is not None and exp_m is not None:
                    ecarts.append(exp_m - cv_c)
                    exp_vals.append(exp_m)
                    cv_vals.append(cv_c)
            if not ecarts:
                continue
            ecart_moy = sum(ecarts) / len(ecarts)
            exp_moy   = sum(exp_vals) / len(exp_vals)
            cv_moy    = sum(cv_vals) / len(cv_vals)
            # Mise en gras du signe si l'écart est significatif (>0.05 en abs)
            signe = f"{ecart_moy:+.4f}"
            if abs(ecart_moy) > 0.05:
                signe = f"**{signe}**"
            out += (f"| {b['modele']} | {section.replace('vs_', '')} | "
                    f"{signe} | {exp_moy:.4f} | {cv_moy:.4f} |\n")

    return out


# ─────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────

def comparer(
    dossier_in:  Path,
    dossier_out: Path,
    ao_filtre:   Optional[str] = None,
):
    print(f"📂 Lecture des benchmarks dans {dossier_in.resolve()}")
    benchmarks = _charger_benchmarks(dossier_in, ao_filtre)
    print(f"   {len(benchmarks)} fichiers de benchmark chargés.\n")

    if not benchmarks:
        return

    # Grouper par AO
    par_ao: Dict[str, List[Dict]] = defaultdict(list)
    for b in benchmarks:
        ao_id = b["ao"].get("id", "?")
        par_ao[ao_id].append(b)

    dossier_out.mkdir(parents=True, exist_ok=True)

    for ao_id, benchs in par_ao.items():
        # Tri stable des modèles par nom pour avoir un ordre cohérent
        benchs.sort(key=lambda b: b["modele"])
        print(f"🔍 AO : {ao_id}  ({len(benchs)} modèles)")
        for b in benchs:
            print(f"     - {b['modele']}")

        # Un sous-dossier par AO pour ne pas mélanger
        dossier_ao = dossier_out / ao_id.replace("/", "_").replace("\\", "_")
        dossier_ao.mkdir(parents=True, exist_ok=True)

        # Tableaux par section et par granularité
        for section in SECTIONS:
            # 1. CV complet
            header, lignes = _construire_tableau_cv_complet(benchs, section)
            base_nom = f"cv_complet_{section}"
            _ecrire_csv(dossier_ao / f"{base_nom}.csv", header, lignes)
            _ecrire_markdown(
                dossier_ao / f"{base_nom}.md",
                f"{ao_id} — CV complet {section}",
                header, lignes
            )

            # 2. Expériences (max)
            header, lignes = _construire_tableau_experiences_max(benchs, section)
            base_nom = f"experiences_max_{section}"
            _ecrire_csv(dossier_ao / f"{base_nom}.csv", header, lignes)
            _ecrire_markdown(
                dossier_ao / f"{base_nom}.md",
                f"{ao_id} — Meilleure expérience {section}",
                header, lignes
            )

            # 3. CV complet vs meilleure expérience (côte à côte + écart)
            header, lignes = _construire_tableau_cv_vs_exp(benchs, section)
            base_nom = f"cv_vs_exp_{section}"
            _ecrire_csv(dossier_ao / f"{base_nom}.csv", header, lignes)
            _ecrire_markdown(
                dossier_ao / f"{base_nom}.md",
                f"{ao_id} — CV complet vs Meilleure expérience ({section})",
                header, lignes
            )

        # Résumé global pour cette AO
        with open(dossier_ao / "RESUME.md", "w", encoding="utf-8") as f:
            f.write(f"# Comparaison des modèles d'embedding — AO {ao_id}\n\n")
            poste = benchs[0]["ao"].get("poste", "")
            if poste:
                f.write(f"**Poste demandé** : {poste}\n\n")
            f.write(_resume_par_modele(benchs))
            f.write("\n\n")
            f.write(_resume_cv_vs_exp(benchs))
            f.write("\n\n## Fichiers détaillés\n\n")
            for section in SECTIONS:
                sec = section.replace('vs_', '')
                f.write(f"- `cv_complet_{section}.md` / `.csv` : score "
                        f"du CV complet vs {sec}\n")
                f.write(f"- `experiences_max_{section}.md` / `.csv` : meilleur "
                        f"score d'expérience vs {sec}\n")
                f.write(f"- `cv_vs_exp_{section}.md` / `.csv` : comparaison "
                        f"côte à côte CV complet / meilleure exp vs {sec}\n")

        print(f"     ✓ tableaux écrits dans {dossier_ao}\n")

    print(f"💾 Comparaisons générées dans {dossier_out.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare les fichiers benchmark_*.json côte à côte."
    )
    parser.add_argument("--input", "-i", default="./benchmark",
                        help="Dossier des fichiers de benchmark (défaut: ./benchmark)")
    parser.add_argument("--output", "-o", default="./benchmark/comparaisons",
                        help="Dossier de sortie (défaut: ./benchmark/comparaisons)")
    parser.add_argument("--ao", default=None,
                        help="Filtrer sur un id d'AO précis (ex: PMU)")
    args = parser.parse_args()

    comparer(
        dossier_in  = Path(args.input),
        dossier_out = Path(args.output),
        ao_filtre   = args.ao,
    )


if __name__ == "__main__":
    main()
