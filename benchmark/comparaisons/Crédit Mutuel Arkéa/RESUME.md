# Comparaison des modèles d'embedding — AO Crédit Mutuel Arkéa

**Poste demandé** : Data Scientist

## Résumé par condition (modèle + instruction)

Calculé sur 'CV complet vs AO complète'. Tri par écart-type décroissant : plus c'est haut, plus la condition discrimine entre bons et mauvais CVs.

| Condition | Nb CV | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-4B | 8 | 0.6029 | **0.0812** | 0.5145 | 0.736 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | 8 | 0.5658 | **0.0734** | 0.4246 | 0.6639 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | 8 | 0.5515 | **0.0727** | 0.4632 | 0.6781 |
| Qwen/Qwen3-Embedding-8B | 8 | 0.5764 | **0.0691** | 0.4576 | 0.6635 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | 8 | 0.5881 | **0.0646** | 0.4664 | 0.6735 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | 8 | 0.5259 | **0.0549** | 0.4505 | 0.6206 |


## CV complet vs Meilleure expérience

Écart moyen = moyenne(exp_max - cv_complet) sur tous les CVs.

- **Positif** : la meilleure expérience est plus discriminante (le CV complet dilue le signal)
- **Négatif** : le CV complet est plus pertinent (le contexte global aide)
- **~0** : les deux stratégies sont équivalentes

| Condition | Section | Écart moyen | exp_max (moy) | cv_complet (moy) |
|---|---|---|---|---|
| Qwen/Qwen3-Embedding-4B | profil | -0.0273 | 0.5727 | 0.6000 |
| Qwen/Qwen3-Embedding-4B | description | -0.0129 | 0.5482 | 0.5612 |
| Qwen/Qwen3-Embedding-4B | contexte | -0.0219 | 0.5205 | 0.5424 |
| Qwen/Qwen3-Embedding-4B | ao_complete | -0.0242 | 0.5787 | 0.6029 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | profil | -0.0212 | 0.4865 | 0.5077 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | description | -0.0142 | 0.4883 | 0.5025 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | contexte | -0.0298 | 0.5165 | 0.5463 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | ao_complete | -0.0171 | 0.5344 | 0.5515 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | profil | -0.0256 | 0.4496 | 0.4752 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | description | -0.0236 | 0.4781 | 0.5018 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | contexte | -0.0326 | 0.4959 | 0.5285 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | ao_complete | -0.0287 | 0.4972 | 0.5259 |
| Qwen/Qwen3-Embedding-8B | profil | -0.0343 | 0.5630 | 0.5972 |
| Qwen/Qwen3-Embedding-8B | description | +0.0113 | 0.5391 | 0.5277 |
| Qwen/Qwen3-Embedding-8B | contexte | +0.0087 | 0.5461 | 0.5374 |
| Qwen/Qwen3-Embedding-8B | ao_complete | -0.0127 | 0.5638 | 0.5764 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | profil | -0.0275 | 0.4839 | 0.5114 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | description | -0.0124 | 0.5203 | 0.5327 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | contexte | -0.0289 | 0.5250 | 0.5539 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | ao_complete | -0.0184 | 0.5474 | 0.5658 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | profil | -0.0170 | 0.4657 | 0.4826 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | description | -0.0189 | 0.4639 | 0.4828 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | contexte | -0.0319 | 0.5176 | 0.5495 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | ao_complete | -0.0264 | 0.5617 | 0.5881 |


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
