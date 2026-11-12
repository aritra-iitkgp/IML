"""
Sequential Covering (RIPPER-style) and Bayesian Rule Lists (BRL) on Adult Census Income.
Core algorithms are written from scratch in PyTorch.

Supporting libraries (not part of the algorithms):
  pandas       : reading the CSV and discretising columns
  numpy        : converting pandas columns into arrays for torch
  scikit-learn : train/test split and evaluation metrics

How PyTorch is used
  Every literal (column = value) becomes a boolean mask over all samples, stored in one
  tensor M of shape (n_literals, n_samples). A rule is the AND of rows of M.
  - FOIL gain is computed for ALL literals at once with tensor ops (no Python loop).
  - BRL candidate mining uses a matrix product to count pair supports.
  - BRL leaf counts and the Beta-binomial likelihood are computed from tensor masks.

Honest naming
  - SequentialCoveringClassifier is RIPPER-STYLE: grow (FOIL gain) + prune (Cohen metric)
    + separate-and-conquer. It has no MDL stopping rule and no optimisation pass.
  - BayesianRuleListClassifier uses simulated annealing (temperature decays 1.0 -> 0.2),
    so it is a MAP-style search, not exact MCMC sampling.
"""

import math
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]


# ======================================================================================
# DATA PREPROCESSING
# ======================================================================================
def load_and_preprocess_adult(file_path="adult.data"):
    df = pd.read_csv(file_path, names=COLUMNS, sep=r',\s*', engine='python',
                     na_values=['?', ' ?'])
    # Check columns explicitly to avoid pandas 3.x object/string dtype warnings.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=['fnlwgt', 'education-num'])
    df['income'] = df['income'].astype(str).str.rstrip('.')

    df['age_group'] = pd.cut(df['age'].astype(float), bins=[0, 25, 45, 65, 120],
                             labels=['Young (<=25)', 'Middle-aged (26-45)',
                                     'Senior-Adult (46-65)', 'Senior (>65)']).astype(str)
    df['hours_group'] = pd.cut(df['hours-per-week'].astype(float), bins=[0, 34, 40, 55, 120],
                               labels=['Part-time (<35)', 'Full-time (35-40)',
                                       'Overtime (41-55)', 'Extreme (>55)']).astype(str)
    df['capital_gain_group'] = pd.cut(df['capital-gain'].astype(float),
                                      bins=[-1, 0, 7000, 1000000],
                                      labels=['None', 'Moderate (<7k)', 'High (>7k)']).astype(str)
    df['capital_loss_group'] = pd.cut(df['capital-loss'].astype(float),
                                      bins=[-1, 0, 1800, 1000000],
                                      labels=['None', 'Low-Loss (<=1.8k)',
                                              'High-Loss (>1.8k)']).astype(str)
    df = df.drop(columns=['age', 'hours-per-week', 'capital-gain', 'capital-loss'])
    df['native_country_group'] = np.where(df['native-country'] == 'United-States',
                                          'United-States', 'Other')
    df = df.drop(columns=['native-country'])

    y = (df['income'] == '>50K').astype(int).values
    return df.drop(columns=['income']), y


# ======================================================================================
# SHARED: literal -> boolean mask encoding (the PyTorch representation)
# ======================================================================================
class _LiteralEncoder:
    """Turns a DataFrame into a bool tensor M of shape (n_literals, n_samples)."""

    def fit_literals(self, X_df):
        self.literals_ = [(col, val) for col in X_df.columns for val in X_df[col].unique()]
        self.lit_col_ = torch.tensor(
            [list(X_df.columns).index(c) for c, _ in self.literals_], dtype=torch.long)
        return self

    def encode(self, X_df):
        cache = {}
        rows = []
        for col, val in self.literals_:
            if col not in cache:
                cache[col] = X_df[col].values
            rows.append(cache[col] == val)
        return torch.from_numpy(np.stack(rows))  # bool (L, N)


def _cover(M, lit_idx):
    """AND of the chosen literal rows. Empty rule covers everything."""
    if len(lit_idx) == 0:
        return torch.ones(M.shape[1], dtype=torch.bool)
    return M[list(lit_idx)].all(dim=0)


# ======================================================================================
# ALGORITHM 1: SEQUENTIAL COVERING (RIPPER-STYLE)
# ======================================================================================
class SequentialCoveringClassifier(_LiteralEncoder):
    """
    1. Separate-and-conquer: learn a rule for the positive class, remove covered rows, repeat.
    2. Grow: greedily add the literal with the best FOIL gain (vectorised over literals).
    3. Prune: drop trailing literals using Cohen's metric (p - n) / (p + n) on a held-out part.
    4. Default rule gives the residual positive rate to uncovered rows.

    laplace=True uses (p+1)/(p+n+2) for rule precision (improvement option).
    """

    def __init__(self, max_rules=10, max_rule_conds=2, prune_fraction=0.33,
                 laplace=False, random_state=42):
        self.max_rules = max_rules
        self.max_rule_conds = max_rule_conds
        self.prune_fraction = prune_fraction
        self.laplace = laplace
        self.random_state = random_state

    # ---- grow ----
    def _grow_rule(self, Mg, yg):
        covered = torch.ones(Mg.shape[1], dtype=torch.bool)
        rule = []
        while len(rule) < self.max_rule_conds:
            p0 = int((covered & yg).sum())
            n0 = int((covered & ~yg).sum())
            if p0 == 0 or n0 == 0:
                break
            C = Mg & covered                                   # (L, Ng) candidate coverage
            p1 = (C & yg).sum(dim=1).double()                  # positives per literal
            n1 = (C & ~yg).sum(dim=1).double()                 # negatives per literal
            prec0 = p0 / (p0 + n0 + 1e-9)
            prec1 = p1 / (p1 + n1 + 1e-9)
            gain = p1 * (torch.log2(prec1.clamp_min(1e-12)) - math.log2(prec0))
            gain[(p1 == 0) | (prec1 <= prec0)] = -1.0
            if rule:
                gain[rule] = -1.0
            best = int(torch.argmax(gain))
            if float(gain[best]) <= 0:
                break
            rule.append(best)
            covered = C[best]
        return rule

    # ---- prune ----
    def _prune_rule(self, rule, Mp, yp):
        def score(lits):
            m = _cover(Mp, lits)
            p = int((m & yp).sum())
            n = int((m & ~yp).sum())
            return ((p - n) / (p + n) if p + n > 0 else -1.0), p, n

        best_score, best_p, best_n = score(rule)
        best_prefix = list(rule)
        for k in range(len(rule) - 1, 0, -1):
            s, p, n = score(rule[:k])
            if s >= best_score:
                best_score, best_p, best_n, best_prefix = s, p, n, rule[:k]
        if best_p <= best_n:
            return None
        return best_prefix

    def fit(self, X_df, y):
        g = torch.Generator().manual_seed(self.random_state)
        self.fit_literals(X_df)
        M = self.encode(X_df)
        yt = torch.from_numpy(np.asarray(y).astype(bool))
        alive = torch.ones(M.shape[1], dtype=torch.bool)

        self.rules_idx_, self.rules_, self.rule_stats_ = [], [], []

        while len(self.rules_) < self.max_rules:
            pos_idx = torch.nonzero(alive & yt).squeeze(1)
            neg_idx = torch.nonzero(alive & ~yt).squeeze(1)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                break
            pos_idx = pos_idx[torch.randperm(len(pos_idx), generator=g)]
            neg_idx = neg_idx[torch.randperm(len(neg_idx), generator=g)]
            pp = max(1, int(len(pos_idx) * self.prune_fraction))
            nn = max(1, int(len(neg_idx) * self.prune_fraction))
            prune_idx = torch.cat([pos_idx[:pp], neg_idx[:nn]])
            grow_idx = torch.cat([pos_idx[pp:], neg_idx[nn:]])

            grown = self._grow_rule(M[:, grow_idx], yt[grow_idx])
            if not grown:
                break
            pruned = self._prune_rule(grown, M[:, prune_idx], yt[prune_idx])
            if not pruned:
                break

            cov = _cover(M, pruned) & alive
            p_full = int((cov & yt).sum())
            n_full = int((cov & ~yt).sum())
            if p_full == 0:
                break
            prec = ((p_full + 1) / (p_full + n_full + 2) if self.laplace
                    else p_full / (p_full + n_full))
            self.rules_idx_.append(pruned)
            self.rules_.append([self.literals_[i] for i in pruned])
            self.rule_stats_.append({'precision': prec, 'pos_covered': p_full,
                                     'neg_covered': n_full,
                                     'total_covered': p_full + n_full})
            alive = alive & ~cov

        n_alive = int(alive.sum())
        self.default_prob_ = float((alive & yt).sum()) / n_alive if n_alive else 0.0
        return self

    def predict_proba(self, X_df):
        M = self.encode(X_df)
        probs = torch.full((M.shape[1],), self.default_prob_, dtype=torch.double)
        uncovered = torch.ones(M.shape[1], dtype=torch.bool)
        for idx, st in zip(self.rules_idx_, self.rule_stats_):
            hit = _cover(M, idx) & uncovered
            probs[hit] = st['precision']
            uncovered &= ~hit
        return probs.numpy()

    def predict(self, X_df, threshold=0.5):
        return (self.predict_proba(X_df) >= threshold).astype(int)

    def print_rules(self):
        print(f"\nSequential Covering rules ({len(self.rules_)}):")
        for i, (rule, st) in enumerate(zip(self.rules_, self.rule_stats_), 1):
            cond = " AND ".join(f"[{c} = '{v}']" for c, v in rule)
            print(f"  Rule {i}: IF {cond} THEN >50K  (precision {st['precision']:.3f}, "
                  f"covers {st['total_covered']} [pos {st['pos_covered']}, "
                  f"neg {st['neg_covered']}])")
        print(f"  DEFAULT: ELSE <=50K  (residual positive rate {self.default_prob_:.3f})\n")


# ======================================================================================
# ALGORITHM 2: BAYESIAN RULE LIST (annealed MAP search)
# ======================================================================================
class BayesianRuleListClassifier(_LiteralEncoder):
    """
    1. Mine candidate rules: frequent singletons and pairs (pair supports via matrix product).
    2. Score a list by log Beta-binomial leaf likelihood + length prior + width penalty.
    3. Search lists with add / remove / swap / replace moves under a decaying temperature.
    4. Leaf probabilities are posterior means (n1 + a1) / (n + a0 + a1).

    n_chains > 1 runs independent searches and keeps the highest-scoring list.
    """

    def __init__(self, list_len_prior=3.0, card_penalty=0.5, alpha0=1.0, alpha1=1.0,
                 minsupport=0.03, n_iter=4000, n_chains=1, random_state=42):
        self.list_len_prior = list_len_prior
        self.card_penalty = card_penalty
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.minsupport = minsupport
        self.n_iter = n_iter
        self.n_chains = n_chains
        self.random_state = random_state

    # ---- candidate mining ----
    def _mine(self, M, min_count):
        counts = M.sum(dim=1)
        sing = torch.nonzero(counts >= min_count).squeeze(1)
        Ms = M[sing].float()
        joint = Ms @ Ms.T                                              # pair supports
        col = self.lit_col_[sing]
        valid = (joint >= min_count) & (col[:, None] != col[None, :])
        valid = torch.triu(valid, diagonal=1)
        pi, pj = torch.nonzero(valid, as_tuple=True)

        self.candidates_ = [(self.literals_[int(i)],) for i in sing]
        masks = [M[sing]]
        self.candidates_ += [(self.literals_[int(sing[a])], self.literals_[int(sing[b])])
                             for a, b in zip(pi.tolist(), pj.tolist())]
        if len(pi):
            masks.append(M[sing[pi]] & M[sing[pj]])
        return torch.cat(masks, dim=0)                                 # (C, N) bool

    def _log_leaf(self, n0, n1):
        a0, a1 = self.alpha0, self.alpha1
        return (math.lgamma(a0 + a1) - math.lgamma(a0) - math.lgamma(a1)
                + math.lgamma(n0 + a0) + math.lgamma(n1 + a1)
                - math.lgamma(n0 + n1 + a0 + a1))

    def _score(self, rule_idx, cm, yt):
        covered = torch.zeros(cm.shape[1], dtype=torch.bool)
        log_lik, probs, counts = 0.0, [], []
        for idx in rule_idx:
            m = cm[idx] & ~covered
            n1, n0 = int((m & yt).sum()), int((m & ~yt).sum())
            counts.append((n0, n1))
            probs.append((n1 + self.alpha1) / (n0 + n1 + self.alpha0 + self.alpha1))
            log_lik += self._log_leaf(n0, n1)
            covered |= m
        d = ~covered
        n1, n0 = int((d & yt).sum()), int((d & ~yt).sum())
        counts.append((n0, n1))
        probs.append((n1 + self.alpha1) / (n0 + n1 + self.alpha0 + self.alpha1))
        log_lik += self._log_leaf(n0, n1)

        m_len = len(rule_idx)
        log_prior = m_len * math.log(self.list_len_prior) - math.lgamma(m_len + 1)
        for idx in rule_idx:
            log_prior -= self.card_penalty * (len(self.candidates_[idx]) - 1)
        return log_lik + log_prior, probs, counts

    def _run_chain(self, cm, yt, seed):
        rng = random.Random(seed)
        n_c = cm.shape[0]

        # initialise with the best single rule, scanning ALL candidates
        best_c, best_s = 0, -1e18
        for i in range(n_c):
            s, _, _ = self._score([i], cm, yt)
            if s > best_s:
                best_c, best_s = i, s

        cur = [best_c]
        cur_s, cur_p, cur_n = self._score(cur, cm, yt)
        best = (list(cur), cur_s, cur_p, cur_n)

        def pick_new(exclude):
            for _ in range(50):
                c = rng.randrange(n_c)
                if c not in exclude:
                    return c
            return None

        for it in range(self.n_iter):
            temp = max(0.2, 1.0 - 0.8 * it / self.n_iter)
            move = rng.randrange(4)
            prop = list(cur)
            if move == 0 and len(prop) < 8:
                c = pick_new(set(cur))
                if c is None:
                    continue
                prop.insert(rng.randint(0, len(prop)), c)
            elif move == 1 and len(prop) > 1:
                prop.pop(rng.randrange(len(prop)))
            elif move == 2 and len(prop) > 1:
                i, j = rng.sample(range(len(prop)), 2)
                prop[i], prop[j] = prop[j], prop[i]
            elif move == 3:
                c = pick_new(set(cur))
                if c is None:
                    continue
                prop[rng.randrange(len(prop))] = c
            else:
                continue

            ps, pp, pn = self._score(prop, cm, yt)
            if rng.random() < math.exp(min(0.0, (ps - cur_s) / temp)):
                cur, cur_s, cur_p, cur_n = prop, ps, pp, pn
                if cur_s > best[1]:
                    best = (list(cur), cur_s, cur_p, cur_n)
        return best

    def fit(self, X_df, y):
        self.fit_literals(X_df)
        M = self.encode(X_df)
        yt = torch.from_numpy(np.asarray(y).astype(bool))
        cm = self._mine(M, int(self.minsupport * M.shape[1]))

        best = None
        for c in range(self.n_chains):
            res = self._run_chain(cm, yt, self.random_state + 1000 * c)
            if best is None or res[1] > best[1]:
                best = res
        self.best_list_, self.best_score_, self.leaf_probs_, self.leaf_counts_ = best
        return self

    def predict_proba(self, X_df):
        M = self.encode(X_df)
        n = M.shape[1]
        probs = torch.full((n,), self.leaf_probs_[-1], dtype=torch.double)
        uncovered = torch.ones(n, dtype=torch.bool)
        for cand_idx, p in zip(self.best_list_, self.leaf_probs_[:-1]):
            lits = [self.literals_.index(l) for l in self.candidates_[cand_idx]]
            hit = _cover(M, lits) & uncovered
            probs[hit] = p
            uncovered &= ~hit
        return probs.numpy()

    def predict(self, X_df, threshold=0.5):
        return (self.predict_proba(X_df) >= threshold).astype(int)

    def print_rules(self):
        print(f"\nBayesian Rule List ({len(self.best_list_)} rules + DEFAULT):")
        for i, (ci, p, cnt) in enumerate(zip(self.best_list_, self.leaf_probs_[:-1],
                                             self.leaf_counts_[:-1]), 1):
            cond = " AND ".join(f"[{c} = '{v}']" for c, v in self.candidates_[ci])
            print(f"  Rule {i}: IF {cond} THEN P(>50K) = {p:.3f}  (n0={cnt[0]}, n1={cnt[1]})")
        print(f"  DEFAULT: ELSE P(>50K) = {self.leaf_probs_[-1]:.3f}  "
              f"(n0={self.leaf_counts_[-1][0]}, n1={self.leaf_counts_[-1][1]})\n")


# ======================================================================================
# SINGLE-RUN EXPERIMENT
# ======================================================================================
def run_experiment(path="adult.data", seed=42):
    X, y = load_and_preprocess_adult(path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                              random_state=seed, stratify=y)
    print(f"Data: {X.shape}, train={len(X_tr)}, test={len(X_te)}")

    models = {
        "Sequential Covering": SequentialCoveringClassifier(random_state=seed),
        "BRL": BayesianRuleListClassifier(random_state=seed),
    }
    rows = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fit_s = time.time() - t0
        model.print_rules()
        prob = model.predict_proba(X_te)
        pred = (prob >= 0.5).astype(int)
        rows[name] = {"Accuracy": accuracy_score(y_te, pred),
                      "Precision": precision_score(y_te, pred, zero_division=0),
                      "Recall": recall_score(y_te, pred, zero_division=0),
                      "F1": f1_score(y_te, pred, zero_division=0),
                      "AUC": roc_auc_score(y_te, prob),
                      "Fit time (s)": fit_s}
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        print(f"{name} confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    run_experiment()