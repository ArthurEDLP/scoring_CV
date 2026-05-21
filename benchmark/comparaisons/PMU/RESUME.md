# Comparaison des modèles d'embedding — AO PMU

**Poste demandé** : Data Analyst / Data Scientist

## Résumé par condition (modèle + instruction)

Calculé sur 'CV complet vs AO complète'. Tri par écart-type décroissant : plus c'est haut, plus la condition discrimine entre bons et mauvais CVs.

| Condition | Nb CV | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-4B | 8 | 0.6161 | **0.0953** | 0.4811 | 0.7808 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | 8 | 0.5409 | **0.0866** | 0.4168 | 0.6853 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | 8 | 0.5842 | **0.0799** | 0.4889 | 0.7341 |
| Qwen/Qwen3-Embedding-8B | 8 | 0.5675 | **0.0777** | 0.4634 | 0.7063 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | 8 | 0.5062 | **0.0447** | 0.4585 | 0.6002 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | 8 | 0.5169 | **0.0336** | 0.4569 | 0.5656 |


## CV complet vs Meilleure expérience

Écart moyen = moyenne(exp_max - cv_complet) sur tous les CVs.

- **Positif** : la meilleure expérience est plus discriminante (le CV complet dilue le signal)
- **Négatif** : le CV complet est plus pertinent (le contexte global aide)
- **~0** : les deux stratégies sont équivalentes

| Condition | Section | Écart moyen | exp_max (moy) | cv_complet (moy) |
|---|---|---|---|---|
| Qwen/Qwen3-Embedding-4B | profil | -0.0301 | 0.5305 | 0.5606 |
| Qwen/Qwen3-Embedding-4B | description | -0.0153 | 0.5789 | 0.5941 |
| Qwen/Qwen3-Embedding-4B | contexte | -0.0010 | 0.4709 | 0.4719 |
| Qwen/Qwen3-Embedding-4B | ao_complete | -0.0358 | 0.5803 | 0.6161 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | profil | -0.0313 | 0.4865 | 0.5178 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | description | -0.0331 | 0.5437 | 0.5768 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | contexte | -0.0250 | 0.5016 | 0.5266 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | ao_complete | -0.0417 | 0.4992 | 0.5409 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | profil | -0.0299 | 0.4317 | 0.4617 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | description | -0.0383 | 0.4803 | 0.5187 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | contexte | -0.0416 | 0.5137 | 0.5553 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | ao_complete | -0.0411 | 0.4652 | 0.5062 |
| Qwen/Qwen3-Embedding-8B | profil | -0.0258 | 0.5398 | 0.5656 |
| Qwen/Qwen3-Embedding-8B | description | +0.0034 | 0.5585 | 0.5551 |
| Qwen/Qwen3-Embedding-8B | contexte | +0.0249 | 0.4828 | 0.4578 |
| Qwen/Qwen3-Embedding-8B | ao_complete | -0.0128 | 0.5547 | 0.5675 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | profil | -0.0366 | 0.5041 | 0.5407 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | description | -0.0269 | 0.5567 | 0.5836 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | contexte | -0.0205 | 0.5180 | 0.5385 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | ao_complete | -0.0271 | 0.5571 | 0.5842 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | profil | -0.0271 | 0.4342 | 0.4613 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | description | -0.0331 | 0.5294 | 0.5625 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | contexte | -0.0269 | 0.5336 | 0.5605 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | ao_complete | -0.0301 | 0.4868 | 0.5169 |


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
