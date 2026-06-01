from pathlib import Path
import matplotlib.pyplot as plt

models = {
    "Qwen_Qwen3-Embedding-4B": {
        "std": 0.0812,
        "values": {
            "JL": 0.7360,
            "SLE": 0.7249,
            "CEU": 0.6021,
            "TVA": 0.5985,
            "MFU": 0.5978,
            "RMA": 0.5293,
            "EMA": 0.5202,
            "KBE": 0.5145,
        },
    },
    "Qwen_Qwen3-Embedding-8B_inst_parcours": {
        "std": 0.0734,
        "values": {
            "SLE": 0.6639,
            "JL": 0.6434,
            "CEU": 0.6063,
            "TVA": 0.6029,
            "MFU": 0.5404,
            "KBE": 0.5281,
            "EMA": 0.5170,
            "RMA": 0.4246,
        },
    },
    "Qwen_Qwen3-Embedding-4B_inst_parcours": {
        "std": 0.0727,
        "values": {
            "JL": 0.6781,
            "SLE": 0.6464,
            "CEU": 0.5665,
            "MFU": 0.5456,
            "TVA": 0.5440,
            "KBE": 0.4849,
            "EMA": 0.4832,
            "RMA": 0.4632,
        },
    },
    "Qwen_Qwen3-Embedding-8B": {
        "std": 0.0691,
        "values": {
            "SLE": 0.6635,
            "JL": 0.6612,
            "TVA": 0.6274,
            "CEU": 0.6035,
            "KBE": 0.5391,
            "MFU": 0.5389,
            "EMA": 0.5203,
            "RMA": 0.4576,
        },
    },
    "Qwen_Qwen3-Embedding-8B_inst_pertinence": {
        "std": 0.0646,
        "values": {
            "SLE": 0.6735,
            "JL": 0.6550,
            "TVA": 0.6325,
            "CEU": 0.6143,
            "MFU": 0.5711,
            "KBE": 0.5596,
            "EMA": 0.5325,
            "RMA": 0.4664,
        },
    },
    "Qwen_Qwen3-Embedding-4B_inst_pertinence": {
        "std": 0.0549,
        "values": {
            "SLE": 0.6206,
            "JL": 0.6013,
            "TVA": 0.5361,
            "KBE": 0.5124,
            "MFU": 0.5111,
            "CEU": 0.4990,
            "EMA": 0.4758,
            "RMA": 0.4505,
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
