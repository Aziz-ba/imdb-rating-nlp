"""Generate the model-comparison chart from results/metrics.json."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parent.parent
metrics = json.loads((root / "results" / "metrics.json").read_text())["models"]

labels = {"tfidf_ridge": "TF-IDF\n+ Ridge", "use_dnn": "USE\n+ DNN", "distilbert": "DistilBERT"}
names = list(labels.values())
mae = [metrics[k]["MAE"] for k in labels]
rmse = [metrics[k]["RMSE"] for k in labels]

x = range(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.2))
b1 = ax.bar([i - w/2 for i in x], mae, w, label="MAE", color="#29B5E8")
b2 = ax.bar([i + w/2 for i in x], rmse, w, label="RMSE", color="#EE4C2C")
ax.set_xticks(list(x)); ax.set_xticklabels(names)
ax.set_ylabel("Error (lower is better)")
ax.set_title("Predicting IMDb rating from plot text — model comparison")
ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = root / "assets" / "model_comparison.png"
fig.savefig(out, dpi=130)
print("wrote", out)
