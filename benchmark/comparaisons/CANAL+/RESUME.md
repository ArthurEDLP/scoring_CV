# Comparaison des modèles d'embedding — AO CANAL+

**Poste demandé** : Consultant ETL & Data Platform

## Résumé par modèle (sur 'CV complet vs AO complète')

Tri par écart-type décroissant : plus c'est haut, plus le modèle discrimine entre bons et mauvais CVs.

| Modèle | Nb CV | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-8B | 7 | 0.6219 | **0.0921** | 0.4845 | 0.7901 |
| Qwen/Qwen3-Embedding-4B | 7 | 0.6033 | **0.0822** | 0.5047 | 0.7607 |
| BAAI/bge-m3 | 7 | 0.5588 | **0.0515** | 0.4763 | 0.6416 |
| intfloat/multilingual-e5-large-instruct | 7 | 0.8733 | **0.0194** | 0.8507 | 0.8965 |


## Fichiers détaillés

- `cv_complet_vs_profil.md` / `.csv` : score du CV complet vs profil
- `experiences_max_vs_profil.md` / `.csv` : meilleur score d'expérience vs profil
- `cv_complet_vs_description.md` / `.csv` : score du CV complet vs description
- `experiences_max_vs_description.md` / `.csv` : meilleur score d'expérience vs description
- `cv_complet_vs_contexte.md` / `.csv` : score du CV complet vs contexte
- `experiences_max_vs_contexte.md` / `.csv` : meilleur score d'expérience vs contexte
- `cv_complet_vs_ao_complete.md` / `.csv` : score du CV complet vs ao_complete
- `experiences_max_vs_ao_complete.md` / `.csv` : meilleur score d'expérience vs ao_complete
