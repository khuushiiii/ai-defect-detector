import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

FEATURES = ["LOC", "Complexity", "CodeChurn", "Commits", "Developers"]
OUTPUT_DIR = "output"


def _save(fig, name: str) -> str:
    """Save a figure to the output directory and close it."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot: {path}")
    return path


def plot_defect_probability(data) -> str:
    """Bar chart: defect probability per module."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = data["Probability"].apply(
        lambda p: "#ff4757" if p > 0.7 else ("#ffa502" if p > 0.4 else "#2ed573")
    )
    ax.bar(data["Module"], data["Probability"], color=colors, edgecolor="none")
    ax.set_title("Defect Probability per Module", fontsize=14, pad=12)
    ax.set_xlabel("Module")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, color="#ff4757", linestyle="--", linewidth=0.8, label="High risk threshold")
    ax.axhline(0.4, color="#ffa502", linestyle="--", linewidth=0.8, label="Medium risk threshold")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right")
    return _save(fig, "defect_probability.png")


def plot_heatmap(data) -> str:
    """Correlation heatmap of all numeric features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric = data[FEATURES + ["Defect"]].corr()
    sns.heatmap(numeric, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    return _save(fig, "heatmap.png")


def plot_feature_importance(model, data) -> str:
    """Horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#00d4ff" if i == importances.argmax() else "#7c3aed" for i in range(len(FEATURES))]
    ax.barh(FEATURES, importances, color=colors, edgecolor="none")
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, pad=12)
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    return _save(fig, "feature_importance.png")