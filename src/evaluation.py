"""Model evaluation pipeline: 60/20/20 splits, validation threshold tuning, and multi-seed benchmarking.

Author: Kotikala Yashwant (24AI10034)
"""
import numpy as np
import pandas as pd

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .data import load_adult, discretise, LiteralEncoder
from .sequential_covering import SequentialCovering
from .bayesian_rule_list import BayesianRuleList


def metrics(y, prob, tau):
    pred = (prob >= tau).astype(int)
    return dict(acc=accuracy_score(y, pred),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, prob))


def best_tau(y_val, prob_val):
    """Threshold maximising minority-class F1 on the validation set."""
    grid = np.unique(np.round(prob_val, 4))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y_val, (prob_val >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def size_sc(model):
    return len(model.rules_), sum(len(r) for r in model.rules_)


def size_brl(model):
    return len(model.best_order_), sum(len(model.cand_[i]) for i in model.best_order_)


def evaluate(file_path="data/adult.data", seeds=(42, 43, 44), brl_iter=4000,
             out_dir="results"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("STAGE 5 — EVALUATION / AUDIT")
    print("=" * 60)
    X, y = load_adult(file_path, verbose=False)
    Xd = discretise(X)
    print(f"Data: {Xd.shape}, positive rate = {y.mean():.4f}")
    maj = max(y.mean(), 1 - y.mean())
    print(f"Majority-class baseline: accuracy={maj:.4f}, minority F1=0\n")

    rows = []
    for s in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(Xd, y, test_size=0.20,
                                                  random_state=s, stratify=y)
        X_sub, X_val, y_sub, y_val = train_test_split(X_tr, y_tr, test_size=0.25,
                                                      random_state=s, stratify=y_tr)
        enc = LiteralEncoder().fit(X_sub)

        sc = SequentialCovering(random_state=s).fit(X_sub, y_sub, enc)
        brl = BayesianRuleList(n_iter=brl_iter, random_state=s).fit(X_sub, y_sub, enc)

        for name, model, size_fn in [("SeqCovering", sc, size_sc),
                                     ("BRL", brl, size_brl)]:
            p_val, p_te = model.predict_proba(X_val), model.predict_proba(X_te)
            tau = best_tau(y_val, p_val)
            n_rules, n_lits = size_fn(model)
            fixed, tuned = metrics(y_te, p_te, 0.5), metrics(y_te, p_te, tau)
            row = dict(model=name, seed=s, rules=n_rules, literals=n_lits, tau=tau)
            row.update({f"fixed_{k}": v for k, v in fixed.items()})
            row.update({f"tuned_{k}": v for k, v in tuned.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "evaluation_raw.csv"), index=False)

    print("=== Per-run results (test set) ===")
    cols = ["model", "seed", "rules", "literals", "tau",
            "fixed_acc", "fixed_f1", "fixed_auc", "tuned_f1", "tuned_rec"]
    print(df[cols].round(4).to_string(index=False))

    print("\n=== Mean +/- std over seeds ===")
    summ = df.groupby("model").agg(["mean", "std"])
    for model in ["SeqCovering", "BRL"]:
        m = summ.loc[model]
        print(f"\n{model}:")
        print(f"  rules     : {m[('rules', 'mean')]:.1f}")
        print(f"  literals  : {m[('literals', 'mean')]:.1f}")
        print(f"  accuracy  : {m[('fixed_acc', 'mean')]:.4f} +/- {m[('fixed_acc', 'std')]:.4f}")
        print(f"  precision : {m[('fixed_prec', 'mean')]:.4f} +/- {m[('fixed_prec', 'std')]:.4f}")
        print(f"  recall    : {m[('fixed_rec', 'mean')]:.4f} +/- {m[('fixed_rec', 'std')]:.4f}")
        print(f"  F1 (0.5)  : {m[('fixed_f1', 'mean')]:.4f} +/- {m[('fixed_f1', 'std')]:.4f}")
        print(f"  F1 (tuned): {m[('tuned_f1', 'mean')]:.4f} +/- {m[('tuned_f1', 'std')]:.4f}")
        print(f"  AUC       : {m[('fixed_auc', 'mean')]:.4f} +/- {m[('fixed_auc', 'std')]:.4f}")

    print("\n=== Accuracy-vs-complexity Pareto (tuned F1 vs literals) ===")
    pts = df.groupby("model").agg(literals=("literals", "mean"),
                                  f1=("tuned_f1", "mean"))
    print(pts.round(4).to_string())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, r in pts.iterrows():
            ax.scatter(r["literals"], r["f1"], s=90,
                       marker="o" if name == "SeqCovering" else "s")
            ax.annotate(name, (r["literals"], r["f1"]), fontsize=9)
        ax.set_xlabel("Explanation size (total literals)")
        ax.set_ylabel("Test F1 (>50K), tuned on validation")
        ax.set_title("Accuracy vs complexity")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "evaluation_pareto.png"), dpi=150)
        print("Saved", os.path.join(out_dir, "evaluation_pareto.png"))
    except Exception as e:
        print("Plot skipped:", e)