"""Metrics shared by training, evaluation, quantization and benchmarking.

Two views of the same predictions:
  - 10-class view: accuracy, macro-F1, per-class recall/precision.
  - Safety view: "is the driver using a phone?" (classes c1-c4 vs the rest).
    Phone recall = of all real phone-use images, how many did we flag.
    A missed phone event is worse than a false alarm, so this is the number
    to protect when choosing a model or a quantization setting.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from data import CLASS_NAMES, NUM_CLASSES, PHONE_CLASSES


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(NUM_CLASSES))
    prec, rec, _, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    phone_true = np.isin(y_true, PHONE_CLASSES)
    phone_pred = np.isin(y_pred, PHONE_CLASSES)
    tp = int((phone_true & phone_pred).sum())
    phone_recall = tp / max(int(phone_true.sum()), 1)
    phone_precision = tp / max(int(phone_pred.sum()), 1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "phone_recall": float(phone_recall),
        "phone_precision": float(phone_precision),
        "per_class_recall": {CLASS_NAMES[i]: float(rec[i]) for i in labels},
        "per_class_precision": {CLASS_NAMES[i]: float(prec[i]) for i in labels},
        "support": {CLASS_NAMES[i]: int(support[i]) for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion_matrix(cm, path, title="Confusion matrix (row-normalized = recall)"):
    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    short = [f"c{i}" for i in range(NUM_CLASSES)]
    ax.set_xticklabels(short)
    ax.set_yticklabels([f"c{i} {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.5 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
