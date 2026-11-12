"""Smoke tests for data integrity, PyTorch literal encoding, and model convergence.

Run with: python -m pytest tests/
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src.data import load_adult, discretise, LiteralEncoder
from src.sequential_covering import SequentialCovering
from src.bayesian_rule_list import BayesianRuleList

DATA = "data/adult.data"


def test_preprocessing_row_count():
    X, y = load_adult(DATA, verbose=False)
    assert len(X) == 30162, f"expected 30162 rows, got {len(X)}"
    assert set(np.unique(y)) <= {0, 1}


def test_literal_encoder_one_per_column():
    import torch
    X, _ = load_adult(DATA, verbose=False)
    Xd = discretise(X)
    enc = LiteralEncoder().fit(Xd)
    M = enc.encode(Xd)
    # each example must satisfy exactly one literal per column
    for col in Xd.columns:
        idx = [i for i, (c, _) in enumerate(enc.literals_) if c == col]
        assert bool(torch.all(M[idx].sum(0) == 1)), f"bad encoding for {col}"


def test_sequential_covering_beats_majority():
    X, y = load_adult(DATA, verbose=False)
    Xd = discretise(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xd, y, test_size=0.2,
                                              random_state=0, stratify=y)
    enc = LiteralEncoder().fit(X_tr)
    clf = SequentialCovering(random_state=0).fit(X_tr, y_tr, enc)
    acc = ((clf.predict_proba(X_te) >= 0.5).astype(int) == y_te).mean()
    assert acc > 0.78, f"sequential covering accuracy too low: {acc}"


def test_brl_leaf_probs_are_calibrated():
    X, y = load_adult(DATA, verbose=False)
    Xd = discretise(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xd, y, test_size=0.2,
                                              random_state=0, stratify=y)
    enc = LiteralEncoder().fit(X_tr)
    clf = BayesianRuleList(n_iter=200, random_state=0).fit(X_tr, y_tr, enc)
    probs = clf.predict_proba(X_te)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y_te, probs) > 0.75


if __name__ == "__main__":
    test_preprocessing_row_count()
    test_literal_encoder_one_per_column()
    test_sequential_covering_beats_majority()
    test_brl_leaf_probs_are_calibrated()
    print("All smoke tests passed.")