from pathlib import Path
import matplotlib.pyplot as plt

models = {
    "Qwen_Qwen3-Embedding-4B": {
        "std": 0.0804,
        "values": {
            "CEU": 0.8071,
            "MFU": 0.7200,
            "JL": 0.6821,
            "SLE": 0.6483,
            "TVA": 0.6204,
            "KBE": 0.5982,
            "RMA": 0.5860,
            "EMA": 0.5342,
        },
    },

    "Qwen_Qwen3-Embedding-8B_inst_parcours": {
        "std": 0.0967,
        "values": {
            "CEU": 0.8207,
            "TVA": 0.6235,
            "JL": 0.6158,
            "SLE": 0.6065,
            "MFU": 0.5969,
            "KBE": 0.5893,
            "EMA": 0.5189,
            "RMA": 0.4632,
        },
    },

    "Qwen_Qwen3-Embedding-4B_inst_parcours": {
        "std": 0.0823,
        "values": {
            "CEU": 0.7187,
            "MFU": 0.6170,
            "JL": 0.5718,
            "SLE": 0.5325,
            "TVA": 0.5263,
            "KBE": 0.5154,
            "RMA": 0.4626,
            "EMA": 0.4450,
        },
    },

    "Qwen_Qwen3-Embedding-8B": {
        "std": 0.0890,
        "values": {
            "CEU": 0.8361,
            "TVA": 0.6730,
            "JL": 0.6657,
            "SLE": 0.6459,
            "KBE": 0.6174,
            "MFU": 0.6160,
            "EMA": 0.5488,
            "RMA": 0.5231,
        },
    },

    "Qwen_Qwen3-Embedding-8B_inst_pertinence": {
        "std": 0.0616,
        "values": {
            "CEU": 0.6793,
            "TVA": 0.5856,
            "JL": 0.5838,
            "KBE": 0.5741,
            "SLE": 0.5731,
            "MFU": 0.5717,
            "EMA": 0.4984,
            "RMA": 0.4560,
        },
    },

    "Qwen_Qwen3-Embedding-4B_inst_pertinence": {
        "std": 0.0337,
        "values": {
            "SLE": 0.5476,
            "KBE": 0.5303,
            "JL": 0.5215,
            "CEU": 0.5201,
            "MFU": 0.5144,
            "TVA": 0.5048,
            "EMA": 0.4518,
            "RMA": 0.4481,
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
