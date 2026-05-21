# Comparaison des modèles d'embedding — AO CANAL+

**Poste demandé** : Consultant ETL & Data Platform

## Résumé par condition (modèle + instruction)

Calculé sur 'CV complet vs AO complète'. Tri par écart-type décroissant : plus c'est haut, plus la condition discrimine entre bons et mauvais CVs.

| Condition | Nb CV | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-8B [inst:parcours] | 7 | 0.5597 | **0.106** | 0.3806 | 0.7587 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | 7 | 0.5106 | **0.0956** | 0.3912 | 0.7079 |
| Qwen/Qwen3-Embedding-8B | 7 | 0.6219 | **0.0921** | 0.4845 | 0.7901 |
| Qwen/Qwen3-Embedding-4B | 7 | 0.6045 | **0.0823** | 0.505 | 0.7624 |
| BAAI/bge-m3 | 7 | 0.5588 | **0.0515** | 0.4763 | 0.6416 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | 7 | 0.5479 | **0.0475** | 0.4602 | 0.6149 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | 7 | 0.473 | **0.0348** | 0.417 | 0.5206 |
| intfloat/multilingual-e5-large-instruct | 7 | 0.8733 | **0.0194** | 0.8507 | 0.8965 |


## CV complet vs Meilleure expérience

Écart moyen = moyenne(exp_max - cv_complet) sur tous les CVs.

- **Positif** : la meilleure expérience est plus discriminante (le CV complet dilue le signal)
- **Négatif** : le CV complet est plus pertinent (le contexte global aide)
- **~0** : les deux stratégies sont équivalentes

| Condition | Section | Écart moyen | exp_max (moy) | cv_complet (moy) |
|---|---|---|---|---|
| BAAI/bge-m3 | profil | -0.0098 | 0.5043 | 0.5142 |
| BAAI/bge-m3 | description | +0.0106 | 0.5071 | 0.4965 |
| BAAI/bge-m3 | contexte | -0.0164 | 0.5476 | 0.5640 |
| BAAI/bge-m3 | ao_complete | -0.0321 | 0.5267 | 0.5588 |
| Qwen/Qwen3-Embedding-4B | profil | **-0.0523** | 0.5088 | 0.5611 |
| Qwen/Qwen3-Embedding-4B | description | -0.0330 | 0.5313 | 0.5643 |
| Qwen/Qwen3-Embedding-4B | contexte | -0.0355 | 0.5508 | 0.5863 |
| Qwen/Qwen3-Embedding-4B | ao_complete | -0.0444 | 0.5601 | 0.6045 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | profil | -0.0498 | 0.4723 | 0.5221 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | description | -0.0359 | 0.4546 | 0.4905 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | contexte | -0.0379 | 0.4218 | 0.4597 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | ao_complete | -0.0464 | 0.4642 | 0.5106 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | profil | -0.0362 | 0.4196 | 0.4558 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | description | -0.0396 | 0.4339 | 0.4735 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | contexte | -0.0369 | 0.4383 | 0.4752 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | ao_complete | -0.0376 | 0.4354 | 0.4730 |
| Qwen/Qwen3-Embedding-8B | profil | **-0.0708** | 0.4785 | 0.5493 |
| Qwen/Qwen3-Embedding-8B | description | -0.0457 | 0.5166 | 0.5623 |
| Qwen/Qwen3-Embedding-8B | contexte | -0.0255 | 0.4988 | 0.5243 |
| Qwen/Qwen3-Embedding-8B | ao_complete | **-0.0540** | 0.5679 | 0.6219 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | profil | **-0.0673** | 0.4724 | 0.5397 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | description | **-0.0539** | 0.4435 | 0.4974 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | contexte | -0.0435 | 0.4665 | 0.5100 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | ao_complete | **-0.0563** | 0.5034 | 0.5597 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | profil | -0.0363 | 0.4221 | 0.4584 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | description | -0.0499 | 0.4518 | 0.5017 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | contexte | -0.0469 | 0.4699 | 0.5168 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | ao_complete | **-0.0543** | 0.4937 | 0.5479 |
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
