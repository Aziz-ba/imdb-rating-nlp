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

indices = list(range(len(names)))
bar_width = 0.38
fig, ax = plt.subplots(figsize=(7, 4.2))
mae_bars = ax.bar([i - bar_width / 2 for i in indices], mae, bar_width, label="MAE", color="#29B5E8")
rmse_bars = ax.bar([i + bar_width / 2 for i in indices], rmse, bar_width, label="RMSE", color="#EE4C2C")
ax.set_xticks(indices)
ax.set_xticklabels(names)
ax.set_ylabel("Error (lower is better)")
ax.set_title("Predicting IMDb rating from plot text — model comparison")
ax.bar_label(mae_bars, fmt="%.3f", padding=2, fontsize=8)
ax.bar_label(rmse_bars, fmt="%.3f", padding=2, fontsize=8)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = root / "assets" / "model_comparison.png"
fig.savefig(out, dpi=130)
print("wrote", out)
