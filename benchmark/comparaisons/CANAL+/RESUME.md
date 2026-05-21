# Comparaison des modèles d'embedding — AO CANAL+

**Poste demandé** : Consultant ETL & Data Platform

## Résumé par condition (modèle + instruction)

Calculé sur 'CV complet vs AO complète'. Tri par écart-type décroissant : plus c'est haut, plus la condition discrimine entre bons et mauvais CVs.

| Condition | Nb CV | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-4B [inst:parcours] | 8 | 0.5394 | **0.1175** | 0.3912 | 0.741 |
| Qwen/Qwen3-Embedding-4B | 8 | 0.6311 | **0.1044** | 0.505 | 0.8177 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | 8 | 0.5698 | **0.1027** | 0.3806 | 0.7587 |
| Qwen/Qwen3-Embedding-8B | 8 | 0.63 | **0.0888** | 0.4845 | 0.7901 |
| BAAI/bge-m3 | 7 | 0.5588 | **0.0515** | 0.4763 | 0.6416 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | 8 | 0.5547 | **0.0479** | 0.4602 | 0.6149 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | 8 | 0.4778 | **0.035** | 0.417 | 0.5206 |
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
| Qwen/Qwen3-Embedding-4B | profil | **-0.0522** | 0.5313 | 0.5835 |
| Qwen/Qwen3-Embedding-4B | description | -0.0241 | 0.5694 | 0.5935 |
| Qwen/Qwen3-Embedding-4B | contexte | -0.0307 | 0.5775 | 0.6082 |
| Qwen/Qwen3-Embedding-4B | ao_complete | -0.0401 | 0.5911 | 0.6311 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | profil | **-0.0503** | 0.4932 | 0.5436 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | description | -0.0250 | 0.4978 | 0.5228 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | contexte | -0.0284 | 0.4606 | 0.4890 |
| Qwen/Qwen3-Embedding-4B [inst:parcours] | ao_complete | -0.0392 | 0.5002 | 0.5394 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | profil | -0.0301 | 0.4264 | 0.4566 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | description | -0.0402 | 0.4404 | 0.4805 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | contexte | -0.0343 | 0.4447 | 0.4791 |
| Qwen/Qwen3-Embedding-4B [inst:pertinence] | ao_complete | -0.0357 | 0.4421 | 0.4778 |
| Qwen/Qwen3-Embedding-8B | profil | **-0.0563** | 0.5054 | 0.5617 |
| Qwen/Qwen3-Embedding-8B | description | -0.0157 | 0.5566 | 0.5723 |
| Qwen/Qwen3-Embedding-8B | contexte | -0.0080 | 0.5185 | 0.5266 |
| Qwen/Qwen3-Embedding-8B | ao_complete | -0.0350 | 0.5950 | 0.6300 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | profil | **-0.0590** | 0.4918 | 0.5508 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | description | -0.0280 | 0.4822 | 0.5102 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | contexte | -0.0258 | 0.4929 | 0.5187 |
| Qwen/Qwen3-Embedding-8B [inst:parcours] | ao_complete | -0.0404 | 0.5295 | 0.5698 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | profil | -0.0269 | 0.4361 | 0.4630 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | description | -0.0437 | 0.4646 | 0.5083 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | contexte | -0.0392 | 0.4820 | 0.5213 |
| Qwen/Qwen3-Embedding-8B [inst:pertinence] | ao_complete | -0.0466 | 0.5082 | 0.5547 |
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
