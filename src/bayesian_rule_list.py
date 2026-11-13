import math
import random

import numpy as np
import torch


class BayesianRuleList:
    def __init__(self, alpha0=1.0, alpha1=1.0, list_prior_lambda=3.0,
                 width_penalty=0.5, max_len=8, minsupport=0.03,
                 n_iter=4000, random_state=42):
        self.alpha0 = alpha0                 # Dirichlet concentration for negatives
        self.alpha1 = alpha1                 # Dirichlet concentration for positives
        self.lam = list_prior_lambda         # truncated-Poisson mean length
        self.width_penalty = width_penalty   # per-extra-condition penalty
        self.max_len = max_len
        self.minsupport = minsupport
        self.n_iter = n_iter
        self.random_state = random_state

    # ---- pre-mining: frequent 1- and 2-literal antecedent candidates ----
    def _mine(self, M, enc, min_count):
        counts = M.sum(1)
        sing = torch.nonzero(counts >= min_count).squeeze(1)
        Ms = M[sing].float()
        joint = Ms @ Ms.T                                   # pair supports (matmul)
        col = enc.literal_col_[sing]
        valid = (joint >= min_count) & (col[:, None] != col[None, :])
        valid = torch.triu(valid, diagonal=1)
        pi, pj = torch.nonzero(valid, as_tuple=True)

        self.cand_ = [(enc.literals_[int(i)],) for i in sing]
        masks = [M[sing]]
        self.cand_ += [(enc.literals_[int(sing[a])], enc.literals_[int(sing[b])])
                       for a, b in zip(pi.tolist(), pj.tolist())]
        if len(pi):
            masks.append(M[sing[pi]] & M[sing[pj]])
        return torch.cat(masks, 0)                          # (C, N) bool

    # ---- log Dirichlet-Multinomial marginal likelihood of one leaf ----
    def _log_leaf(self, n0, n1):
        a0, a1 = self.alpha0, self.alpha1
        return (math.lgamma(a0 + a1) - math.lgamma(a0) - math.lgamma(a1)
                + math.lgamma(n0 + a0) + math.lgamma(n1 + a1)
                - math.lgamma(n0 + n1 + a0 + a1))

    # ---- log posterior of a list, plus posterior-mean leaf probabilities ----
    def _log_posterior(self, order, cm, y):
        covered = torch.zeros(cm.shape[1], dtype=torch.bool)
        loglik, probs, counts = 0.0, [], []
        for idx in order:
            m = cm[idx] & ~covered
            n1, n0 = int((m & y).sum()), int((m & ~y).sum())
            counts.append((n0, n1))
            probs.append((n1 + self.alpha1) / (n0 + n1 + self.alpha0 + self.alpha1))
            loglik += self._log_leaf(n0, n1)
            covered |= m
        d = ~covered
        n1, n0 = int((d & y).sum()), int((d & ~y).sum())
        counts.append((n0, n1))                              # default leaf
        probs.append((n1 + self.alpha1) / (n0 + n1 + self.alpha0 + self.alpha1))
        loglik += self._log_leaf(n0, n1)

        m_len = len(order)
        # truncated-Poisson prior on list length: lam^m / m!
        log_prior = m_len * math.log(self.lam) - math.lgamma(m_len + 1)
        for idx in order:
            log_prior -= self.width_penalty * (len(self.cand_[idx]) - 1)
        return loglik + log_prior, probs, counts

    # ---- Metropolis-Hastings search over lists ----
    def _mh(self, cm, y, seed):
        rng = random.Random(seed)
        n_c = cm.shape[0]

        best_c, best_s = 0, -1e18
        for i in range(n_c):
            s, _, _ = self._log_posterior([i], cm, y)
            if s > best_s:
                best_c, best_s = i, s
        cur = [best_c]
        cur_s, cur_p, cur_n = self._log_posterior(cur, cm, y)
        best = (list(cur), cur_s, cur_p, cur_n)

        def other(exclude):
            for _ in range(50):
                c = rng.randrange(n_c)
                if c not in exclude:
                    return c
            return None

        for _ in range(self.n_iter):
            move = rng.randrange(4)
            prop = list(cur)
            q_forward = 0.0
            q_backward = 0.0

            # We approximate proposal probabilities by counting how many
            # valid moves of each type exist in the current and proposed
            # states, so add/remove and replace moves are treated as
            # non-symmetric in the MH ratio.
            cur_set = set(cur)

            if move == 0 and len(prop) < self.max_len:          # add
                c = other(cur_set)
                if c is None:
                    continue
                # number of possible forward add moves: candidates not in cur
                n_fwd = max(len(set(range(n_c)) - cur_set), 1)
                # number of positions to insert
                pos = rng.randint(0, len(prop))
                prop.insert(pos, c)
                q_forward = 1.0 / n_fwd
                # backward (remove) moves from prop: any position can be removed
                q_backward = 1.0 / len(prop)
            elif move == 1 and len(prop) > 1:                   # remove
                # forward: choose a position to remove
                idx = rng.randrange(len(prop))
                removed = prop.pop(idx)
                q_forward = 1.0 / len(cur)
                # backward (add) from prop: can add any candidate not in prop
                prop_set = set(prop)
                n_bwd = max(len(set(range(n_c)) - prop_set), 1)
                q_backward = 1.0 / n_bwd
            elif move == 2 and len(prop) > 1:                   # swap order
                i, j = rng.sample(range(len(prop)), 2)
                prop[i], prop[j] = prop[j], prop[i]
                # swapping two positions is symmetric
                q_forward = 1.0
                q_backward = 1.0
            elif move == 3:                                     # replace
                # forward: replace one element of cur with a different candidate
                idx = rng.randrange(len(prop))
                c = other(cur_set)
                if c is None:
                    continue
                old = prop[idx]
                prop[idx] = c
                # forward: choose position *and* new candidate
                n_fwd_pos = len(cur)
                n_fwd_cand = max(len(set(range(n_c)) - cur_set), 1)
                q_forward = 1.0 / (n_fwd_pos * n_fwd_cand)
                # backward: from prop, we can replace the same position back to old
                prop_set = set(prop)
                n_bwd_pos = len(prop)
                n_bwd_cand = max(len(set(range(n_c)) - prop_set) + 1, 1)
                q_backward = 1.0 / (n_bwd_pos * n_bwd_cand)
            else:
                continue

            ps, pp, pn = self._log_posterior(prop, cm, y)
            # Metropolis-Hastings with non-symmetric proposals:
            # accept with probability min(1, exp(ps - cur_s) * (q_backward / q_forward)).
            if q_forward > 0 and q_backward > 0:
                log_alpha = (ps - cur_s) + math.log(q_backward) - math.log(q_forward)
                if math.log(rng.random() + 1e-12) < log_alpha:
                    cur, cur_s, cur_p, cur_n = prop, ps, pp, pn
                    if cur_s > best[1]:
                        best = (list(cur), cur_s, cur_p, cur_n)
        return best

    def fit(self, X_df, y, enc):
        self._enc = enc
        M = enc.encode(X_df)
        y = torch.from_numpy(np.asarray(y).astype(bool))
        self.cand_masks_ = self._mine(M, enc, int(self.minsupport * M.shape[1]))
        best = self._mh(self.cand_masks_, y, self.random_state)
        self.best_order_, self.best_score_, self.leaf_probs_, self.leaf_counts_ = best
        return self

    def predict_proba(self, X_df):
        M = self._enc.encode(X_df)
        probs = torch.full((M.shape[1],), self.leaf_probs_[-1], dtype=torch.double)
        uncovered = torch.ones(M.shape[1], dtype=torch.bool)
        for cidx, p in zip(self.best_order_, self.leaf_probs_[:-1]):
            # Coverage of a rule is defined by its literals, so it is rebuilt
            # against the input passed in (not the training-time mined masks).
            lits = [self._enc.literals_.index(l) for l in self.cand_[cidx]]
            hit = self._enc.rule_mask(M, lits) & uncovered
            probs[hit] = p
            uncovered &= ~hit
        return probs.numpy()

    def print_rules(self):
        print(f"\nBayesian Rule List ({len(self.best_order_)} rules + DEFAULT):")
        for i, (ci, p, cnt) in enumerate(zip(self.best_order_, self.leaf_probs_[:-1],
                                             self.leaf_counts_[:-1]), 1):
            cond = " AND ".join(f"[{c} = '{v}']" for c, v in self.cand_[ci])
            print(f"  Rule {i}: IF {cond} THEN P(>50K) = {p:.3f} "
                  f"(n0={cnt[0]}, n1={cnt[1]})")
        print(f"  DEFAULT: ELSE P(>50K) = {self.leaf_probs_[-1]:.3f} "
              f"(n0={self.leaf_counts_[-1][0]}, n1={self.leaf_counts_[-1][1]})\n")
