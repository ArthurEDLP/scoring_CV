from pathlib import Path
import matplotlib.pyplot as plt

models = {
    "Qwen_Qwen3-Embedding-4B": {
        "std": 0.0953,
        "values": {
            "SLE": 0.7808,
            "JL": 0.7283,
            "MFU": 0.6413,
            "TVA": 0.6153,
            "CEU": 0.6002,
            "RMA": 0.5768,
            "EMA": 0.5053,
            "KBE": 0.4811,
        },
    },

    "Qwen_Qwen3-Embedding-8B_inst_parcours": {
        "std": 0.0799,
        "values": {
            "SLE": 0.7341,
            "CEU": 0.6438,
            "JL": 0.6232,
            "TVA": 0.6047,
            "MFU": 0.5776,
            "EMA": 0.5062,
            "KBE": 0.4950,
            "RMA": 0.4889,
        },
    },

    "Qwen_Qwen3-Embedding-4B_inst_parcours": {
        "std": 0.0866,
        "values": {
            "SLE": 0.6853,
            "JL": 0.6388,
            "MFU": 0.5719,
            "CEU": 0.5523,
            "TVA": 0.5363,
            "RMA": 0.4845,
            "EMA": 0.4413,
            "KBE": 0.4168,
        },
    },

    "Qwen_Qwen3-Embedding-8B": {
        "std": 0.0777,
        "values": {
            "SLE": 0.7063,
            "JL": 0.6352,
            "TVA": 0.5992,
            "CEU": 0.5905,
            "MFU": 0.5636,
            "RMA": 0.5055,
            "EMA": 0.4766,
            "KBE": 0.4634,
        },
    },

    "Qwen_Qwen3-Embedding-8B_inst_pertinence": {
        "std": 0.0336,
        "values": {
            "SLE": 0.5656,
            "TVA": 0.5416,
            "MFU": 0.5381,
            "JL": 0.5274,
            "CEU": 0.5215,
            "KBE": 0.5101,
            "EMA": 0.4740,
            "RMA": 0.4569,
        },
    },

    "Qwen_Qwen3-Embedding-4B_inst_pertinence": {
        "std": 0.0447,
        "values": {
            "SLE": 0.6002,
            "JL": 0.5439,
            "MFU": 0.5095,
            "KBE": 0.5059,
            "TVA": 0.5013,
            "CEU": 0.4705,
            "EMA": 0.4602,
            "RMA": 0.4585,
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
