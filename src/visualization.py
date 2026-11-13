
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def _box(ax, x, y, w, h, text, color):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           linewidth=1.2, edgecolor="black", facecolor=color)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, wrap=True)


def plot_decision_list(rows, title, path):
    """Draw rows of (condition, outcome) from top to bottom."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(12, max(3, 1.05 * n + 1)))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, n + 0.5)
    ax.axis("off")
    for i, (condition, outcome) in enumerate(rows):
        y = n - i - 0.5
        is_default = i == n - 1
        fill = "#f3f4f6" if is_default else ("#dbeafe" if i % 2 == 0 else "#fee2e2")
        label = ("ELSE" if is_default else f"Rule {i + 1}") + "\n" + condition
        _box(ax, 0.06, y, 0.70, 0.70, label, fill)
        ax.text(0.80, y + 0.35, outcome, ha="left", va="center", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def decision_list_rows_sc(model, enc):
    rows = []
    for idx, stats in zip(model.rules_, model.stats_):
        rows.append((f"IF {enc.text(idx)}",
                     f"P(>50K) = {stats['precision']:.3f}"))
    rows.append((f"residual P(>50K) = {model.default_prob_:.3f}",
                 "DEFAULT"))
    return rows


def decision_list_rows_brl(model):
    rows = []
    for cidx, prob in zip(model.best_order_, model.leaf_probs_[:-1]):
        condition = " AND ".join(f"{c} = {v}" for c, v in model.cand_[cidx])
        rows.append((f"IF {condition}", f"P(>50K) = {prob:.3f}"))
    rows.append((f"P(>50K) = {model.leaf_probs_[-1]:.3f}", "DEFAULT"))
    return rows


def plot_roc_curves(curves, path):
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, y_true, probability in curves:
        fpr, tpr, _ = roc_curve(y_true, probability)
        auc = roc_auc_score(y_true, probability)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (>50K)")
    ax.set_title("ROC curves on the held-out test set", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion_matrices(models, y_true, tau, path):
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, probability) in zip(axes, models):
        matrix = confusion_matrix(y_true, (probability >= tau).astype(int))
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{name} (threshold {tau:.2f})", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["<=50K", ">50K"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["<=50K", ">50K"])
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(j, i, str(value), ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if value > matrix.max() / 2 else "black")
    fig.suptitle("Confusion matrices on the held-out test set", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_calibration(models, y_true, path, n_bins=8):
    """Draw mean predicted probability against observed positive frequency."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    bins = np.linspace(0, 1, n_bins + 1)
    for name, probability in models:
        bin_index = np.digitize(probability, bins) - 1
        means, observed = [], []
        for b in range(n_bins):
            selected = bin_index == b
            if selected.sum() >= 10:
                means.append(probability[selected].mean())
                observed.append(y_true[selected].mean())
        ax.plot(means, observed, "o-", linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Mean predicted P(>50K)")
    ax.set_ylabel("Observed frequency of >50K")
    ax.set_title("Calibration / reliability diagram", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
