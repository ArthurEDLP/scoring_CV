"""
Backend FastAPI pour le matching CV/AO (usage LOCAL, mono-utilisateur).

Démarrage :
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --reload --port 8000
    puis ouvrir http://localhost:8000

Routes :
    GET    /api/state            -> contenu des dossiers (brutes + prêts)
    POST   /api/upload/{kind}    -> dépose des .json dans *_brutes (kind=cv|ao)
    DELETE /api/ao/{ao_id}       -> supprime une AO (brute + traité + cache global)
    POST   /api/prepare          -> job: prétraitement + caches + score global
    POST   /api/match            -> job: lance THE_HONORED_ONE pour une AO
    GET    /api/jobs/{job_id}     -> état/progression/résultat d'un job
    GET    /logo.png             -> logo (si présent à la racine)
"""

from __future__ import annotations

import os
import json
import sys
import uuid
import subprocess
import threading
import traceback
from pathlib import Path
from typing import Dict, List

from score_global import chemin_cache_global

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse

# ─────────────────────────── Dossiers ─────────────────────────────────────
RACINE        = Path(__file__).parent
AO_BRUTES     = RACINE / "AO_JSON_brutes"
CV_BRUTES     = RACINE / "CV_JSON_brutes"
AO_TRAITES    = RACINE / "AO_JSON"
CV_TRAITES    = RACINE / "CV_JSON"
CACHE_GLOBAL  = RACINE / "cache_global"
for d in (AO_BRUTES, CV_BRUTES, AO_TRAITES, CV_TRAITES, CACHE_GLOBAL):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Matching CV/AO")

# ─────────────────────── Registre de jobs (mémoire) ───────────────────────
JOBS: Dict[str, Dict] = {}
_LOCK = threading.Lock()


def _nouveau_job() -> str:
    jid = uuid.uuid4().hex[:12]
    with _LOCK:
        JOBS[jid] = {"statut": "en_cours", "etape": "", "progression": 0.0,
                     "message": "", "resultat": None, "erreur": None}
    return jid


def _maj_job(jid: str, **kw) -> None:
    with _LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


# ─────────────────────────── Helpers fichiers ─────────────────────────────
def _ids_dans(dossier: Path) -> List[str]:
    return sorted(p.stem for p in dossier.glob("*.json"))


def _lire_id_interne(chemin: Path) -> str:
    try:
        data = json.loads(chemin.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("id"):
            return str(data["id"])
    except Exception:
        pass
    return chemin.stem


# ════════════════════════════ ROUTES API ══════════════════════════════════

@app.get("/api/state")
def etat():
    aos_prets = []
    for p in sorted(AO_TRAITES.glob("*.json")):
        ao_id = _lire_id_interne(p)
        cache = chemin_cache_global(ao_id, str(CACHE_GLOBAL)).exists()
        aos_prets.append({"id": ao_id, "fichier": p.stem, "score_global": cache})

    cache_cv_dir = RACINE / "cache_cv"
    cvs_prets = []
    for p in sorted(CV_TRAITES.glob("*.json")):
        cv_id = _lire_id_interne(p)
        # Même normalisation que CacheEmbeddingsCV._chemin
        safe_id = cv_id.replace("/", "_").replace("\\", "_")
        embedding_pret = (cache_cv_dir / f"{safe_id}.json").exists()
        cvs_prets.append({"id": cv_id, "fichier": p.stem,
                          "embedding_pret": embedding_pret})

    return {
        "ao_brutes":  _ids_dans(AO_BRUTES),
        "cv_brutes":  _ids_dans(CV_BRUTES),
        "aos_prets":  aos_prets,
        "cvs_prets":  cvs_prets,
    }


@app.post("/api/upload/{kind}")
async def upload(kind: str, fichiers: List[UploadFile] = File(...)):
    if kind not in ("cv", "ao"):
        raise HTTPException(400, "kind doit être 'cv' ou 'ao'")
    cible = CV_BRUTES if kind == "cv" else AO_BRUTES
    deposes = []
    for f in fichiers:
        if not f.filename.lower().endswith(".json"):
            continue
        contenu = await f.read()
        try:
            json.loads(contenu)
        except Exception:
            raise HTTPException(400, f"{f.filename} n'est pas un JSON valide")
        (cible / Path(f.filename).name).write_bytes(contenu)
        deposes.append(f.filename)
    return {"deposes": deposes}


@app.delete("/api/ao/{ao_id}")
def supprimer_ao(ao_id: str):
    cibles = [
        AO_BRUTES  / f"{ao_id}.json",
        AO_TRAITES / f"{ao_id}.json",
        chemin_cache_global(ao_id, str(CACHE_GLOBAL)),
    ]
    supprimes = [c.name for c in cibles if c.exists()]
    for c in cibles:
        c.unlink(missing_ok=True)
    if not supprimes:
        raise HTTPException(404, f"Rien à supprimer pour {ao_id}")
    return {"supprimes": supprimes}


@app.post("/api/prepare")
def preparer(background: BackgroundTasks):
    jid = _nouveau_job()
    background.add_task(_job_preparer, jid)
    return {"job_id": jid}


@app.post("/api/match")
def matcher(payload: Dict, background: BackgroundTasks):
    offre_id = (payload or {}).get("offre_id")
    if not offre_id:
        raise HTTPException(400, "offre_id manquant")
    jid = _nouveau_job()
    background.add_task(_job_matcher, jid, offre_id)
    return {"job_id": jid}


@app.get("/api/jobs/{job_id}")
def etat_job(job_id: str):
    with _LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job inconnu")
    return job


# ════════════════════════════ JOBS (tâche de fond) ════════════════════════

def _run_tolerant(cmd: List[str]) -> str:
    """Comme _run mais ne lève PAS sur code != 0 : retourne la sortie pour log."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    res = subprocess.run(cmd, cwd=str(RACINE), capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env)
    return (res.stdout or "") + (res.stderr or "")


def _job_preparer(jid: str) -> None:
    """3 étapes : prétraitement -> caches embeddings -> score global."""
    try:
        py = sys.executable
        ao_files = [str(p) for p in AO_BRUTES.glob("*.json")]
        cv_files = [str(p) for p in CV_BRUTES.glob("*.json")]

        _maj_job(jid, etape="Prétraitement des AO", progression=0.05)
        if ao_files:
            _run_tolerant([py, "pretraiter_ao.py", "--output", str(AO_TRAITES), "--force", *ao_files])

        _maj_job(jid, etape="Prétraitement des CV", progression=0.20)
        if cv_files:
            _run_tolerant([py, "pretraiter_cv.py", "--output", str(CV_TRAITES), "--force", *cv_files])

        # Imports lourds ici seulement
        _maj_job(jid, etape="Construction des caches d'embeddings", progression=0.35)
        import THE_HONORED_ONE as pipe
        cvs    = pipe.charger_cvs("./CV_JSON")
        offres = pipe.charger_offres("./AO_JSON")
        total  = max(1, len(cvs) + len(offres))
        fait   = 0
        for o in offres:
            pipe.CACHE_OFFRE.obtenir(o); fait += 1
            _maj_job(jid, progression=0.35 + 0.35 * fait / total)
        for c in cvs:
            pipe.CACHE_CV.obtenir(c); fait += 1
            _maj_job(jid, progression=0.35 + 0.35 * fait / total)

        _maj_job(jid, etape="Calcul du score global (Qwen3-8B)", progression=0.72)
        from precalcul_global import precalculer_ao
        for i, o in enumerate(offres, 1):
            precalculer_ao(o, cvs, dossier="./cache_global")
            _maj_job(jid, progression=0.72 + 0.27 * i / max(1, len(offres)))

        _maj_job(jid, statut="termine", etape="Terminé", progression=1.0,
                 message=f"{len(cvs)} CV · {len(offres)} AO prêts")
    except Exception as e:
        _maj_job(jid, statut="erreur", erreur=str(e),
                 message=traceback.format_exc()[-1200:])


def _job_matcher(jid: str, offre_id: str) -> None:
    try:
        _maj_job(jid, etape=f"Matching pour {offre_id}", progression=0.2)
        import THE_HONORED_ONE as pipe
        resultat = pipe.lancer_matching(offre_id)
        _maj_job(jid, statut="termine", etape="Terminé", progression=1.0,
                 resultat=resultat)
    except Exception as e:
        _maj_job(jid, statut="erreur", erreur=str(e),
                 message=traceback.format_exc()[-1200:])


# ════════════════════════════ FRONT ═══════════════════════════════════════

@app.get("/logo.png")
def logo():
    for nom in ("logo_consort.png", "logo.png"):
        p = RACINE / nom
        if p.exists():
            return FileResponse(str(p))
    raise HTTPException(404, "logo absent")


@app.get("/", response_class=HTMLResponse)
def index():
    chemin = RACINE / "index.html"
    if not chemin.exists():
        return HTMLResponse("<h1>index.html introuvable</h1>", status_code=500)
    return HTMLResponse(chemin.read_text(encoding="utf-8"))