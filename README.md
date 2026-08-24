# Matching CV / AO

Outil local de *matching* entre des CV et des appels d'offres (AO), basé sur des
embeddings sémantiques. Interface web (FastAPI + page HTML unique), calculs
d'embeddings délégués à **Ollama** en local. Usage mono-utilisateur, sur ta
machine — aucune donnée n'est envoyée à l'extérieur.

---

## Structure du projet

```
.
├── app.py                  API FastAPI + service de la page web (POINT DE DÉMARRAGE)
├── Graph.py                Pipeline de matching (graphe LangGraph) — CŒUR du système
├── scoring.py              Catégorisation, score technos, bonus entreprise, durées
├── guards.py               Garde-fous : cohérence du CV + récence du parcours
├── embedding_cache.py      Cache d'embeddings CV/AO (sérialisation, hash, dates)
├── score_global.py         Construction des textes + embeddings Ollama + cosinus
├── precalcul_global.py     Précalcul de l'indicateur global par AO
├── pretraiter_ao.py        AO brute -> AO structurée (via LLM)
├── pretraiter_cv.py        CV brut  -> CV nettoyé (déterministe, sans LLM)
├── CV_AO_Loader.py         Chargement des JSON depuis les dossiers
├── State.py                Schéma d'état du graphe LangGraph
├── taxonomie.py            Familles de technos + compatibilité
├── index.html              Interface web (page unique)
├── logo_consort.png        Logo affiché dans l'interface
│
├── config.json             Seuils de durée d'expérience (en mois)
├── paliers.json            Paliers de séniorité affichés
├── familles_technos.json   Taxonomie technos + paires de familles compatibles
├── requirements.txt        Dépendances Python
├── .gitignore
├── README.md
│
├── AO_JSON_brutes/         [entrée]   AO déposées telles quelles (upload)
├── CV_JSON_brutes/         [entrée]   CV déposés tels quels (upload)
│
├── AO_JSON/                [généré]   AO structurées par pretraiter_ao.py
├── CV_JSON/                [généré]   CV nettoyés par pretraiter_cv.py
│
├── cache_offre/            [généré]   Embeddings des AO (recalcul si contenu modifié)
├── cache_cv/               [généré]   Embeddings des CV
├── cache_global/           [généré]   Cosinus précalculés CV complet ↔ AO complète
│
├── benchmark/              Scripts de benchmark des modèles (hors application ;
│                           dépendances propres, cf. §6)
└── __pycache__/            [généré]   Bytecode Python (ignoré par git)
```

> **[entrée]** = tu y déposes tes fichiers · **[généré]** = créé automatiquement
> par l'application, ne pas éditer à la main. Les dossiers `[généré]` peuvent être
> supprimés sans risque : ils seront reconstruits à la prochaine préparation.

---

## 1. Prérequis

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installé et lancé (c'est lui qui calcule les
  embeddings et fait tourner l'extraction LLM). Le code Python **ne fonctionne
  pas sans Ollama.**
- **GPU NVIDIA recommandé.** Le modèle d'embedding par défaut est le **8B**
  (~6-7 Go de VRAM). Sur une machine sans GPU suffisant, préfère la variante 4B
  (voir §6).

---

## 2. Installer et préparer Ollama

Une fois Ollama installé, laisse-le tourner dans un terminal dédié :

```bash
ollama serve
```

Puis, dans un autre terminal, télécharge les **deux** modèles utilisés par le
projet :

```bash
ollama pull qwen3-embedding:8b     # embeddings (cœur du matching)
ollama pull qwen2.5:7b-instruct    # extraction structurée des AO (pretraiter_ao.py)
```

> Vérification rapide qu'Ollama répond :
> ```bash
> curl http://localhost:11434/api/tags
> ```

---

## 3. Installer le projet

```bash
git clone <URL_DU_REPO>
cd <dossier_du_repo>

# Environnement virtuel (recommandé)
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 4. Lancer l'application

Depuis la **racine du dépôt** (important : les scripts utilisent des chemins
relatifs comme `./CV_JSON`) :

```bash
uvicorn app:app --port 8000
```

Puis ouvre **http://localhost:8000** dans ton navigateur.

`--reload` peut être ajouté pendant le développement pour recharger à chaud.

---

## 5. Utilisation (workflow)

Tout se fait depuis la page web, dans cet ordre :

1. **Déposer les fichiers.** Uploade tes CV et tes AO au format JSON brut
   (1 fichier = 1 CV ou 1 AO). Ils atterrissent dans `CV_JSON_brutes/` et
   `AO_JSON_brutes/`.

2. **Préparer** (bouton *Préparer*). Cette étape enchaîne automatiquement :
   - le prétraitement des AO (via `qwen2.5:7b-instruct`) et des CV,
   - la construction des caches d'embeddings (CV + AO),
   - le **précalcul de l'indicateur global** (cosinus CV complet ↔ AO complète).

   C'est l'étape longue (elle appelle Ollama pour chaque CV et chaque AO).
   À relancer chaque fois que tu ajoutes / modifies des CV ou des AO — le cache
   se recalcule uniquement pour les fichiers dont le contenu a changé (hash MD5).

3. **Matcher.** Sélectionne une AO, lance le matching, et consulte le classement
   des CV par catégorie (groupe principal / profils alternatifs), avec les CV
   écartés par les garde-fous et leur motif.

> **Précalcul en ligne de commande (optionnel).** L'étape *Préparer* fait déjà
> le précalcul global. Si tu veux le relancer seul :
> ```bash
> python precalcul_global.py                 # toutes les AO de ./AO_JSON
> python precalcul_global.py --only <AO_ID>  # une seule AO
> ```

---

## 6. Configuration

| Fichier                 | Rôle                                                                 |
|-------------------------|---------------------------------------------------------------------|
| `config.json`           | Seuils de durée d'expérience en **mois** (`seuil_court_mois`, `seuil_valide_mois`). Éditable depuis l'interface. |
| `paliers.json`          | Paliers de séniorité affichés. Recréé avec des valeurs par défaut s'il est absent. |
| `familles_technos.json` | Taxonomie des technos + paires de familles compatibles (gate sémantique du scoring). |

**Changer de modèle d'embedding** (ex. passer du 8B au 4B) : le nom du modèle
est défini à **deux endroits** qu'il faut garder synchronisés —
`score_global.py` et `Graph.py` (constante `MODEL`). Pense aussi à
`ollama pull qwen3-embedding:4b` au préalable.

**Benchmark.** Le dossier `benchmark/` contient les scripts d'évaluation des
modèles ; ils ne sont pas nécessaires pour faire tourner l'application et ont
leurs propres dépendances (PyTorch, transformers, scikit-learn…), non incluses
dans `requirements.txt`.

---

## 7. Dépannage

- **`ModuleNotFoundError`** au lancement → l'environnement virtuel n'est pas
  activé, ou `pip install -r requirements.txt` n'a pas été fait.
- **« Ollama injoignable » / erreur de connexion `localhost:11434`** → le serveur
  Ollama n'est pas lancé (`ollama serve`), ou les modèles ne sont pas téléchargés
  (§2).
- **Matching qui échoue sur l'indicateur global** (« cache absent pour l'AO … »)
  → relance l'étape *Préparer*, ou `python precalcul_global.py --only <AO_ID>`.
- **Lancer les scripts en dehors de la racine** casse les chemins relatifs
  (`./CV_JSON`, `./cache_global`, …). Reste toujours à la racine du dépôt.