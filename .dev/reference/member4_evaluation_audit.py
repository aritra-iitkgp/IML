"""
Member 4: iterative evaluation harness (RIPPER-style sequential covering vs BRL).

Everything printed here is COMPUTED. No hardcoded results.

Protocol
  - For each seed: stratified split into train_sub (60%), val (20%), test (20%).
  - Every model is fit on train_sub only.
  - tau = 0.5 results ("fixed") and tau tuned on val ("tuned") are both reported on test.
  - Test set is never used to choose anything.
  - Mean +/- std over seeds.

Iterations (each one = hypothesis -> change -> measured result)
  0  baseline            : configs as in the current code, tau = 0.5
  1  threshold tuning    : same models, tau chosen on validation (Member 4's own lever)
  2  RIPPER sweep        : max_rules x max_rule_conds
  3  BRL sweep           : list_len_prior x minsupport
  Pareto plot: test F1 vs number of literals (explanation size).

Usage:  python member4_evaluation_audit.py --data adult.data --seeds 3
"""
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from adult_census_rule_models import (load_and_preprocess_adult,
                                      SequentialCoveringClassifier,
                                      BayesianRuleListClassifier)


# ----------------------------------------------------------------------------- helpers
def metrics(y, prob, tau):
    pred = (prob >= tau).astype(int)
    return {
        "acc": accuracy_score(y, pred),
        "prec": precision_score(y, pred, zero_division=0),
        "rec": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob),
    }


def best_tau(y_val, prob_val):
    """Threshold maximising minority-class F1 on the validation set."""
    grid = np.unique(np.round(prob_val, 4))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y_val, (prob_val >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def size_of(model):
    """(#rules, #literals) of the learned rule model."""
    if hasattr(model, "rules_"):
        return len(model.rules_), sum(len(r) for r in model.rules_)
    return (len(model.best_list_),
            sum(len(model.candidates_[i]) for i in model.best_list_))


def make_splits(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                              random_state=seed, stratify=y)
    X_sub, X_val, y_sub, y_val = train_test_split(X_tr, y_tr, test_size=0.25,
                                                  random_state=seed, stratify=y_tr)
    return X_sub, X_val, X_te, y_sub, y_val, y_te


def run_config(name, factory, X, y, seeds):
    rows = []
    for s in seeds:
        X_sub, X_val, X_te, y_sub, y_val, y_te = make_splits(X, y, s)
        t0 = time.time()
        model = factory(s).fit(X_sub, y_sub)
        fit_s = time.time() - t0
        p_val, p_te = model.predict_proba(X_val), model.predict_proba(X_te)
        tau = best_tau(y_val, p_val)
        n_rules, n_lits = size_of(model)
        fixed, tuned = metrics(y_te, p_te, 0.5), metrics(y_te, p_te, tau)
        row = {"config": name, "seed": s, "tau": tau, "rules": n_rules,
               "literals": n_lits, "fit_s": fit_s}
        row.update({f"fixed_{k}": v for k, v in fixed.items()})
        row.update({f"tuned_{k}": v for k, v in tuned.items()})
        rows.append(row)
        print(f"  [{name}] seed={s} fixedF1={fixed['f1']:.3f} "
              f"tunedF1={tuned['f1']:.3f} tau={tau:.3f} "
              f"rules={n_rules} lits={n_lits} ({fit_s:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    cols = ["fixed_prec", "fixed_rec", "fixed_f1", "fixed_auc",
            "tuned_prec", "tuned_rec", "tuned_f1", "rules", "literals", "fit_s"]
    g = df.groupby("config", sort=False)[cols]
    m, s = g.mean(), g.std(ddof=0)
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = [f"{a:.3f}±{b:.3f}" if c not in ("rules", "literals", "fit_s")
                  else f"{a:.1f}" for a, b in zip(m[c], s[c])]
    return out, m


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="adult.data")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--brl_iter", type=int, default=4000)
    ap.add_argument("--out", default="member4_results")
    a = ap.parse_args()
    seeds = list(range(42, 42 + a.seeds))

    X, y = load_and_preprocess_adult(a.data)
    print(f"Data: {X.shape}, positive rate = {y.mean():.4f}")

    # Majority-class baseline, computed (not typed in)
    maj = max(y.mean(), 1 - y.mean())
    print(f"Majority-class baseline: accuracy={maj:.4f}, minority recall=0, F1=0\n")

    R = lambda mr, mc: (lambda s: SequentialCoveringClassifier(
        max_rules=mr, max_rule_conds=mc, random_state=s))
    B = lambda lam, ms: (lambda s: BayesianRuleListClassifier(
        list_len_prior=lam, card_penalty=0.5, minsupport=ms,
        n_iter=a.brl_iter, random_state=s))

    all_runs = []
    print("== Iteration 0/1: baseline configs (fixed tau=0.5 vs tau tuned on val) ==")
    all_runs.append(run_config("RIPPER mr=10,mc=2", R(10, 2), X, y, seeds))
    all_runs.append(run_config("BRL lam=3,sup=.03", B(3.0, 0.03), X, y, seeds))

    print("\n== Iteration 2: RIPPER complexity sweep ==")
    for mr, mc in [(5, 2), (5, 3), (10, 3)]:
        all_runs.append(run_config(f"RIPPER mr={mr},mc={mc}", R(mr, mc), X, y, seeds))

    print("\n== Iteration 3: BRL prior / support sweep ==")
    for lam, ms in [(2.0, 0.03), (4.0, 0.03), (6.0, 0.03), (3.0, 0.05)]:
        all_runs.append(run_config(f"BRL lam={lam:g},sup={ms:g}", B(lam, ms), X, y, seeds))

    df = pd.concat(all_runs, ignore_index=True)
    df.to_csv(f"{a.out}_raw.csv", index=False)
    table, means = summarize(df)
    table.to_csv(f"{a.out}_table.csv")
    print("\n================ MASTER ITERATIVE COMPARISON (test set, mean±std) ================")
    print(table.to_string())

    # Pareto frontier: tuned F1 vs literals (smaller & better = non-dominated)
    pts = means[["literals", "tuned_f1"]].copy()
    front = []
    for n, r in pts.iterrows():
        dominated = ((pts["literals"] <= r["literals"]) & (pts["tuned_f1"] >= r["tuned_f1"]) &
                     ((pts["literals"] < r["literals"]) | (pts["tuned_f1"] > r["tuned_f1"]))).any()
        if not dominated:
            front.append(n)
    print("\nPareto-optimal configs (tuned F1 vs literals):", front)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for n, r in pts.iterrows():
            ax.scatter(r["literals"], r["tuned_f1"],
                       marker="o" if n.startswith("RIPPER") else "s",
                       s=90 if n in front else 40)
            ax.annotate(n, (r["literals"], r["tuned_f1"]), fontsize=7)
        ax.set_xlabel("Explanation size (total literals)")
        ax.set_ylabel("Test F1 (>50K), tau tuned on val")
        ax.set_title("Accuracy-vs-complexity Pareto view")
        fig.tight_layout()
        fig.savefig(f"{a.out}_pareto.png", dpi=150)
        print(f"Saved {a.out}_pareto.png")
    except Exception as e:  # plotting is optional
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()