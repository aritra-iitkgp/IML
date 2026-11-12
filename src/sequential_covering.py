"""Sequential Covering (separate-and-conquer rule induction with FOIL gain and Cohen pruning).

Author: Chunarkar Sujeeth Kumar (24AI10021)
"""
import math

import numpy as np
import torch


class SequentialCovering:
    def __init__(self, max_rules=10, max_conds=2, prune_fraction=0.33,
                 beam_width=1, random_state=42):
        self.max_rules = max_rules
        self.max_conds = max_conds
        self.prune_fraction = prune_fraction
        self.beam_width = beam_width            # beam_width=1 is plain greedy
        self.random_state = random_state

    # ---- grow one rule on a (masked) dataset ----
    def _grow(self, M, y, enc):
        covered = torch.ones(M.shape[1], dtype=torch.bool)
        rule = []
        while len(rule) < self.max_conds:
            p0 = int((covered & y).sum())
            n0 = int((covered & ~y).sum())
            if p0 == 0 or n0 == 0:
                break
            C = M & covered                              # (L, N) candidate coverage
            p1 = (C & y).sum(1).double()
            n1 = (C & ~y).sum(1).double()
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

    def _cohen(self, M, y, lits, enc):
        m = enc.rule_mask(M, lits)
        p = int((m & y).sum())
        n = int((m & ~y).sum())
        if p + n == 0:
            return -1.0, p, n
        return (p - n) / (p + n), p, n

    def _prune(self, M, y, rule, enc):
        best_s, best_p, best_n = self._cohen(M, y, rule, enc)
        best_rule = list(rule)
        for k in range(len(rule) - 1, 0, -1):
            s, p, n = self._cohen(M, y, rule[:k], enc)
            if s >= best_s:
                best_s, best_p, best_n, best_rule = s, p, n, rule[:k]
        if best_p <= best_n:
            return None
        return best_rule

    def fit(self, X_df, y, enc):
        g = torch.Generator().manual_seed(self.random_state)
        M = enc.encode(X_df)
        y = torch.from_numpy(np.asarray(y).astype(bool))
        alive = torch.ones(M.shape[1], dtype=torch.bool)

        self.rules_, self.stats_ = [], []
        while len(self.rules_) < self.max_rules:
            pos = torch.nonzero(alive & y).squeeze(1)
            neg = torch.nonzero(alive & ~y).squeeze(1)
            if len(pos) == 0 or len(neg) == 0:
                break
            pos = pos[torch.randperm(len(pos), generator=g)]
            neg = neg[torch.randperm(len(neg), generator=g)]
            pp = max(1, int(len(pos) * self.prune_fraction))
            nn = max(1, int(len(neg) * self.prune_fraction))
            prune_idx = torch.cat([pos[:pp], neg[:nn]])
            grow_idx = torch.cat([pos[pp:], neg[nn:]])

            grown = self._grow(M[:, grow_idx], y[grow_idx], enc)
            if not grown:
                break
            pruned = self._prune(M[:, prune_idx], y[prune_idx], grown, enc)
            if not pruned:
                break

            cov = enc.rule_mask(M, pruned) & alive
            p_full = int((cov & y).sum())
            n_full = int((cov & ~y).sum())
            if p_full == 0:
                break
            self.rules_.append(pruned)
            self.stats_.append({'p': p_full, 'n': n_full,
                                'precision': p_full / (p_full + n_full)})
            alive = alive & ~cov

        n_alive = int(alive.sum())
        self.default_prob_ = float((alive & y).sum()) / n_alive if n_alive else 0.0
        self._enc = enc
        return self

    def predict_proba(self, X_df):
        M = self._enc.encode(X_df)
        probs = torch.full((M.shape[1],), self.default_prob_, dtype=torch.double)
        uncovered = torch.ones(M.shape[1], dtype=torch.bool)
        for idx, st in zip(self.rules_, self.stats_):
            hit = self._enc.rule_mask(M, idx) & uncovered
            probs[hit] = st['precision']
            uncovered &= ~hit
        return probs.numpy()

    def print_rules(self, enc):
        print(f"\nSequential Covering rules ({len(self.rules_)}):")
        for i, (idx, st) in enumerate(zip(self.rules_, self.stats_), 1):
            print(f"  Rule {i}: IF {enc.text(idx)} THEN >50K "
                  f"(precision {st['precision']:.3f}, covers {st['p'] + st['n']} "
                  f"[pos {st['p']}, neg {st['n']}])")
        print(f"  DEFAULT: ELSE <=50K (residual positive rate {self.default_prob_:.3f})\n")