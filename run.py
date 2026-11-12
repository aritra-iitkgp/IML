"""Command-line entry point for the project.

Runs a single stage of the pipeline:

  Stage 1 : data loading, cleaning, and trivial baselines      (Aritra Bose)
  Stage 2 : interpretable discretisation + literal encoding    (Aritra Bose)
  Stage 3 : Sequential Covering (RIPPER-style)                 (Chunarkar Sujeeth Kumar)
  Stage 4 : Bayesian Rule List                                 (Mohammad Asad ur Rahman Nisar)
  Stage 5 : evaluation / audit and comparison                  (Kotikala Yashwant)
  Stage 6 : decision-list and evaluation visualisations         (Kotikala Yashwant)

Usage:  python run.py --stage N
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB

from src.data import load_adult, discretise, LiteralEncoder
from src.sequential_covering import SequentialCovering
from src.bayesian_rule_list import BayesianRuleList
from src.evaluation import evaluate
from src.visualization import (decision_list_rows_brl, decision_list_rows_sc,
                               ensure_dir, plot_calibration,
                               plot_confusion_matrices, plot_decision_list,
                               plot_roc_curves)

DATA = "data/adult.data"


def stage1(file_path=DATA):
    print("=" * 60)
    print("STAGE 1 — DATA + BASELINES")
    print("=" * 60)
    X, y = load_adult(file_path)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                              random_state=42, stratify=y)

    print("\nMAJORITY CLASS BASELINE")
    maj_pred = np.zeros_like(y_te)
    print(f"  Accuracy : {accuracy_score(y_te, maj_pred):.4f}   "
          f"(predicts <=50K for everything)")
    print(f"  Precision/Recall/F1 on >50K : "
          f"{precision_score(y_te, maj_pred, zero_division=0):.3f} / "
          f"{recall_score(y_te, maj_pred, zero_division=0):.3f} / "
          f"{f1_score(y_te, maj_pred, zero_division=0):.3f}")

    print("\nNAIVE BAYES BASELINE (categorical, includes ALL columns incl. raw numerics)")

    def fit_codes(frame):
        return {col: {v: i for i, v in enumerate(pd.unique(frame[col]))}
                for col in frame.columns}

    def apply_codes(frame, codes):
        out = np.zeros((len(frame), len(frame.columns)), dtype=int)
        for j, col in enumerate(frame.columns):
            cats = codes[col]
            # Map unseen (test-only) categories to an existing code.
            out[:, j] = [cats.get(v, 0) for v in frame[col]]
        return out

    codes = fit_codes(X_tr)
    nb = CategoricalNB().fit(apply_codes(X_tr, codes), y_tr)
    prob = nb.predict_proba(apply_codes(X_te, codes))[:, 1]
    pred = (prob >= 0.5).astype(int)
    print(f"  Accuracy  : {accuracy_score(y_te, pred):.4f}")
    print(f"  Precision : {precision_score(y_te, pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_te, pred, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(y_te, pred, zero_division=0):.4f}")
    print(f"  AUC       : {roc_auc_score(y_te, prob):.4f}")


def stage2(file_path=DATA):
    import torch
    print("=" * 60)
    print("STAGE 2 — DISCRETISATION + LITERAL ENCODING")
    print("=" * 60)
    X, y = load_adult(file_path, verbose=False)
    Xd = discretise(X)
    print(f"Features after discretisation : {X.shape[1]} -> {Xd.shape[1]}")
    print(f"Categorical feature columns   : {list(Xd.columns)}")

    enc = LiteralEncoder().fit(Xd)
    M = enc.encode(Xd)
    print(f"Total literals (column=value) : {len(enc.literals_)}")
    print(f"Coverage tensor M             : {tuple(M.shape)}  dtype={M.dtype}")

    per_col = sum(M[[i for i, (c, _) in enumerate(enc.literals_) if c == col]].sum(0)
                  for col in Xd.columns[:3])
    print(f"Sanity: first 3 columns each cover every example exactly once -> "
          f"{bool(torch.all(per_col == 3))}")


def stage3(file_path=DATA, seed=42):
    print("=" * 60)
    print("STAGE 3 — SEQUENTIAL COVERING")
    print("=" * 60)
    X, y = load_adult(file_path, verbose=False)
    Xd = discretise(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xd, y, test_size=0.20,
                                              random_state=seed, stratify=y)

    enc = LiteralEncoder().fit(X_tr)
    clf = SequentialCovering(random_state=seed).fit(X_tr, y_tr, enc)
    clf.print_rules(enc)

    prob = clf.predict_proba(X_te)
    pred = (prob >= 0.5).astype(int)
    print("Sequential Covering test metrics:")
    print(f"  Accuracy  : {accuracy_score(y_te, pred):.4f}")
    print(f"  Precision : {precision_score(y_te, pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_te, pred, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(y_te, pred, zero_division=0):.4f}")
    print(f"  AUC       : {roc_auc_score(y_te, prob):.4f}")


def stage4(file_path=DATA, seed=42):
    print("=" * 60)
    print("STAGE 4 — BAYESIAN RULE LIST")
    print("=" * 60)
    X, y = load_adult(file_path, verbose=False)
    Xd = discretise(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xd, y, test_size=0.20,
                                              random_state=seed, stratify=y)

    enc = LiteralEncoder().fit(X_tr)
    clf = BayesianRuleList(random_state=seed).fit(X_tr, y_tr, enc)
    clf.print_rules()

    prob = clf.predict_proba(X_te)
    pred = (prob >= 0.5).astype(int)
    print("BRL test metrics:")
    print(f"  Accuracy  : {accuracy_score(y_te, pred):.4f}")
    print(f"  Precision : {precision_score(y_te, pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_te, pred, zero_division=0):.4f}")
    print(f"  F1        : {f1_score(y_te, pred, zero_division=0):.4f}")
    print(f"  AUC       : {roc_auc_score(y_te, prob):.4f}")


def stage6(file_path=DATA, seed=42):
    """Generate report figures from one reproducible 80/20 test split."""
    print("=" * 60)
    print("STAGE 6 — VISUALISATIONS")
    print("=" * 60)
    X, y = load_adult(file_path, verbose=False)
    Xd = discretise(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xd, y, test_size=0.20,
                                              random_state=seed, stratify=y)
    enc = LiteralEncoder().fit(X_tr)
    sc = SequentialCovering(random_state=seed).fit(X_tr, y_tr, enc)
    brl = BayesianRuleList(random_state=seed).fit(X_tr, y_tr, enc)
    sc_prob = sc.predict_proba(X_te)
    brl_prob = brl.predict_proba(X_te)

    # Naive Bayes is included only in the ROC comparison as an interpretable baseline.
    codes = {col: {v: i for i, v in enumerate(pd.unique(X_tr[col]))}
             for col in X_tr.columns}
    def encode(frame):
        return np.asarray([[codes[col].get(value, 0) for col, value in row.items()]
                          for _, row in frame.iterrows()], dtype=int)
    nb = CategoricalNB().fit(encode(X_tr), y_tr)
    nb_prob = nb.predict_proba(encode(X_te))[:, 1]

    output = "results/figures"
    ensure_dir(output)
    plot_decision_list(decision_list_rows_sc(sc, enc),
                       "Sequential Covering decision list",
                       f"{output}/sequential_covering_decision_list.png")
    plot_decision_list(decision_list_rows_brl(brl),
                       "Bayesian Rule List decision list",
                       f"{output}/bayesian_rule_list_decision_list.png")
    plot_roc_curves([("Sequential Covering", y_te, sc_prob),
                     ("Bayesian Rule List", y_te, brl_prob),
                     ("Naive Bayes", y_te, nb_prob)],
                    f"{output}/roc_curves.png")
    plot_confusion_matrices([("Sequential Covering", sc_prob),
                             ("Bayesian Rule List", brl_prob)],
                            y_te, 0.5, f"{output}/confusion_matrices.png")
    plot_calibration([("Sequential Covering", sc_prob),
                      ("Bayesian Rule List", brl_prob)],
                     y_te, f"{output}/calibration.png")
    for name in ["sequential_covering_decision_list.png",
                 "bayesian_rule_list_decision_list.png", "roc_curves.png",
                 "confusion_matrices.png", "calibration.png"]:
        print(f"Saved {output}/{name}")


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4,
          5: lambda path=DATA: evaluate(path),
          6: stage6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=sorted(STAGES))
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    STAGES[a.stage](a.data)


if __name__ == "__main__":
    main()