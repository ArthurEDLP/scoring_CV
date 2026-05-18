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


## CV complet vs Meilleure expérience

Écart moyen = moyenne(exp_max - cv_complet) sur tous les CVs.

- **Positif** : la meilleure expérience est plus discriminante (le CV complet dilue le signal)
- **Négatif** : le CV complet est plus pertinent (le contexte global aide)
- **~0** : les deux stratégies sont équivalentes

| Modèle | Section | Écart moyen | exp_max (moy) | cv_complet (moy) |
|---|---|---|---|---|
| BAAI/bge-m3 | profil | -0.0098 | 0.5043 | 0.5142 |
| BAAI/bge-m3 | description | +0.0106 | 0.5071 | 0.4965 |
| BAAI/bge-m3 | contexte | -0.0164 | 0.5476 | 0.5640 |
| BAAI/bge-m3 | ao_complete | -0.0321 | 0.5267 | 0.5588 |
| Qwen/Qwen3-Embedding-4B | profil | **-0.0527** | 0.5108 | 0.5635 |
| Qwen/Qwen3-Embedding-4B | description | -0.0327 | 0.5280 | 0.5608 |
| Qwen/Qwen3-Embedding-4B | contexte | -0.0356 | 0.5500 | 0.5856 |
| Qwen/Qwen3-Embedding-4B | ao_complete | -0.0446 | 0.5587 | 0.6033 |
| Qwen/Qwen3-Embedding-8B | profil | **-0.0708** | 0.4785 | 0.5493 |
| Qwen/Qwen3-Embedding-8B | description | -0.0457 | 0.5166 | 0.5623 |
| Qwen/Qwen3-Embedding-8B | contexte | -0.0255 | 0.4988 | 0.5243 |
| Qwen/Qwen3-Embedding-8B | ao_complete | **-0.0540** | 0.5679 | 0.6219 |
| intfloat/multilingual-e5-large-instruct | profil | +0.0020 | 0.8696 | 0.8676 |
| intfloat/multilingual-e5-large-instruct | description | -0.0159 | 0.8531 | 0.8691 |
| intfloat/multilingual-e5-large-instruct | contexte | -0.0159 | 0.8621 | 0.8780 |
| intfloat/multilingual-e5-large-instruct | ao_complete | -0.0174 | 0.8559 | 0.8733 |


## Fichiers détaillés

- `cv_complet_vs_profil.md` / `.csv` : score du CV complet vs profil
- `experiences_max_vs_profil.md` / `.csv` : meilleur score d'expérience vs profil
- `cv_vs_exp_vs_profil.md` / `.csv` : comparaison côte à côte CV complet / meilleure exp vs profil
- `cv_complet_vs_description.md` / `.csv` : score du CV complet vs description
- `experiences_max_vs_description.md` / `.csv` : meilleur score d'expérience vs description
- `cv_vs_exp_vs_description.md` / `.csv` : comparaison côte à côte CV complet / meilleure exp vs description
- `cv_complet_vs_contexte.md` / `.csv` : score du CV complet vs contexte
- `experiences_max_vs_contexte.md` / `.csv` : meilleur score d'expérience vs contexte
- `cv_vs_exp_vs_contexte.md` / `.csv` : comparaison côte à côte CV complet / meilleure exp vs contexte
- `cv_complet_vs_ao_complete.md` / `.csv` : score du CV complet vs ao_complete
- `experiences_max_vs_ao_complete.md` / `.csv` : meilleur score d'expérience vs ao_complete
- `cv_vs_exp_vs_ao_complete.md` / `.csv` : comparaison côte à côte CV complet / meilleure exp vs ao_complete
