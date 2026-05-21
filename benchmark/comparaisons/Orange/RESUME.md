# Comparaison des modèles d'embedding — AO Orange

**Poste demandé** : Data Engineer

## Résumé par condition (modèle + instruction)

Calculé sur 'CV complet vs AO complète'. Tri par écart-type décroissant : plus c'est haut, plus la condition discrimine entre bons et mauvais CVs.

| Condition | Nb CV | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-8B [inst:parcours] | 8 | 0.6044 | **0.0967** | 0.4632 | 0.8207 |
| Qwen/Qwen3-Embedding-8B | 8 | 0.6407 | **0.089** | 0.5231 | 0.8361 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | 8 | 0.5487 | **0.0823** | 0.445 | 0.7187 |
| Qwen/Qwen3-Embedding-4B | 8 | 0.6495 | **0.0804** | 0.5342 | 0.8071 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | 8 | 0.5653 | **0.0616** | 0.456 | 0.6793 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | 8 | 0.5048 | **0.0337** | 0.4481 | 0.5476 |


## CV complet vs Meilleure expérience

Écart moyen = moyenne(exp_max - cv_complet) sur tous les CVs.

- **Positif** : la meilleure expérience est plus discriminante (le CV complet dilue le signal)
- **Négatif** : le CV complet est plus pertinent (le contexte global aide)
- **~0** : les deux stratégies sont équivalentes

| Condition | Section | Écart moyen | exp_max (moy) | cv_complet (moy) |
|---|---|---|---|---|
| Qwen/Qwen3-Embedding-4B | profil | **-0.0573** | 0.5499 | 0.6072 |
| Qwen/Qwen3-Embedding-4B | description | -0.0423 | 0.5906 | 0.6328 |
| Qwen/Qwen3-Embedding-4B | contexte | -0.0161 | 0.5540 | 0.5702 |
| Qwen/Qwen3-Embedding-4B | ao_complete | **-0.0559** | 0.5937 | 0.6495 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | profil | -0.0438 | 0.4842 | 0.5280 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | description | -0.0458 | 0.5012 | 0.5470 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | contexte | -0.0467 | 0.5217 | 0.5685 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | ao_complete | -0.0407 | 0.5080 | 0.5487 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | profil | -0.0345 | 0.4420 | 0.4765 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | description | **-0.0554** | 0.4869 | 0.5423 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | contexte | -0.0482 | 0.4721 | 0.5203 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | ao_complete | -0.0347 | 0.4701 | 0.5048 |
| Qwen/Qwen3-Embedding-8B | profil | -0.0321 | 0.5723 | 0.6044 |
| Qwen/Qwen3-Embedding-8B | description | -0.0134 | 0.5871 | 0.6005 |
| Qwen/Qwen3-Embedding-8B | contexte | -0.0007 | 0.5896 | 0.5904 |
| Qwen/Qwen3-Embedding-8B | ao_complete | -0.0272 | 0.6135 | 0.6407 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | profil | -0.0334 | 0.5084 | 0.5418 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | description | -0.0424 | 0.5378 | 0.5802 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | contexte | -0.0432 | 0.5591 | 0.6023 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | ao_complete | -0.0266 | 0.5778 | 0.6044 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | profil | -0.0246 | 0.4526 | 0.4772 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | description | -0.0387 | 0.4776 | 0.5162 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | contexte | -0.0451 | 0.5156 | 0.5607 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | ao_complete | -0.0387 | 0.5266 | 0.5653 |


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
