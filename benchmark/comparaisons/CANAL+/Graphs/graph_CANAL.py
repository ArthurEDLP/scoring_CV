from pathlib import Path
import matplotlib.pyplot as plt

models = {
    "Qwen_Qwen3-Embedding-4B": {
        "std": 0.1044,
        "values": {
            "MFU": 0.8177,
            "CEU": 0.7624,
            "JL": 0.6641,
            "SLE": 0.6227,
            "TVA": 0.5984,
            "KBE": 0.5506,
            "RMA": 0.5283,
            "EMA": 0.5050,
        },
    },

    "Qwen_Qwen3-Embedding-8B_inst_parcours": {
        "std": 0.1027,
        "values": {
            "CEU": 0.7587,
            "MFU": 0.6407,
            "TVA": 0.6069,
            "JL": 0.5671,
            "SLE": 0.5670,
            "KBE": 0.5439,
            "EMA": 0.4939,
            "RMA": 0.3806,
        },
    },

    "Qwen_Qwen3-Embedding-4B_inst_parcours": {
        "std": 0.1175,
        "values": {
            "MFU": 0.7410,
            "CEU": 0.7079,
            "JL": 0.5570,
            "TVA": 0.5207,
            "SLE": 0.4992,
            "KBE": 0.4709,
            "EMA": 0.4276,
            "RMA": 0.3912,
        },
    },

    "Qwen_Qwen3-Embedding-8B": {
        "std": 0.0888,
        "values": {
            "CEU": 0.7901,
            "MFU": 0.6874,
            "TVA": 0.6635,
            "JL": 0.6556,
            "SLE": 0.6332,
            "KBE": 0.5969,
            "EMA": 0.5292,
            "RMA": 0.4845,
        },
    },

    "Qwen_Qwen3-Embedding-8B_inst_pertinence": {
        "std": 0.0479,
        "values": {
            "CEU": 0.6149,
            "MFU": 0.6024,
            "TVA": 0.5826,
            "KBE": 0.5617,
            "JL": 0.5568,
            "SLE": 0.5568,
            "EMA": 0.5026,
            "RMA": 0.4602,
        },
    },

    "Qwen_Qwen3-Embedding-4B_inst_pertinence": {
        "std": 0.0350,
        "values": {
            "SLE": 0.5206,
            "KBE": 0.5153,
            "MFU": 0.5113,
            "JL": 0.4828,
            "TVA": 0.4714,
            "CEU": 0.4643,
            "EMA": 0.4395,
            "RMA": 0.4170,
        },
    },
}

output_dir = Path("C:/Users/a.ernoul-delaprovote/OneDrive - CONSORT NT/Bureau/Learn IA/Agents/scoring_CV/benchmark/comparaisons")
output_dir.mkdir(exist_ok=True)

generated_files = []

for model_name, model_data in models.items():
    std = model_data["std"]

    # Sort values descending
    sorted_items = sorted(
        model_data["values"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    labels = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    errors = [std] * len(values)

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        labels,
        values,
        yerr=errors,
        fmt='o-',
        capsize=5
    )

    plt.xlabel("CV")
    plt.ylabel("Score")
    plt.title(f"{model_name} — scores décroissants avec écart-type")
    plt.ylim(0.3, 0.85)
    plt.grid(True)

    file_path = output_dir / f"{model_name}.png"
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()

    generated_files.append(str(file_path))

print("Fichiers générés :")
for f in generated_files:
    print(f)
