# Interpretable Rule Learning in PyTorch: Sequential Covering and Bayesian Rule Lists

**Course:** Interpretable Machine Learning  
**Group Project:** Implementation of inherently interpretable rule learners from scratch in PyTorch.

### Team Members
- Aritra Bose (25EC65R08)
- Chunarkar Sujeeth Kumar (24AI10021)
- Mohammad Asad ur Rahman Nisar (24AI10038)
- Kotikala Yashwant (24AI10034)

---

## Overview

This repository contains clean from-scratch PyTorch implementations of two rule-based classification models evaluated on the UCI Adult Census Income dataset (`adult.data`):

1. **Sequential Covering:** Separate-and-conquer rule induction using FOIL information gain for greedy rule growth and Cohen's metric for rule pruning.
2. **Bayesian Rule Lists (BRL):** Probabilistic decision list learning combining Dirichlet-Multinomial leaf marginal likelihoods with a truncated Poisson length prior, sampled using the Metropolis-Hastings MCMC algorithm.

Core algorithms are implemented directly in PyTorch and standard Python without any third-party rule-learning packages. Supporting libraries (`pandas`, `scikit-learn`, `matplotlib`) are used solely for data ingestion, baseline benchmarks, standard evaluation metrics, and visualization.

---

## Environment Setup

Create and activate a virtual environment, then install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the Pipeline

Execution is organized into modular stages through `run.py`:

```bash
python run.py --stage 1   # Load dataset, inspect statistics, and run baselines
python run.py --stage 2   # Run discretization and verify PyTorch literal tensor encoding
python run.py --stage 3   # Train and evaluate Sequential Covering on an 80/20 split
python run.py --stage 4   # Train and evaluate Bayesian Rule List on an 80/20 split
python run.py --stage 5   # Multi-seed 60/20/20 evaluation with validation threshold tuning
python run.py --stage 6   # Generate decision lists, ROC, confusion matrix, and calibration figures
```

Stages 5 and 6 save outputs directly into `results/`:
- `results/evaluation_raw.csv`: Raw multi-seed evaluation metrics across seeds 42, 43, 44.
- `results/evaluation_pareto.png`: Pareto frontier comparing explanation size (literals) vs. tuned test F1-score.
- `results/figures/`: Decision list diagrams, ROC curves, confusion matrices, and calibration curves.

---

## Project Structure

```text
.
├── run.py                 # Command-line entry point for running stages 1 through 6
├── report.md              # Complete project report and experimental findings
├── requirements.txt       # Pinned library dependencies
├── data/
│   ├── adult.data         # UCI Adult Census Income training data
│   ├── adult.names        # Attribute documentation and metadata
│   └── adult.test         # Raw test split
├── src/
│   ├── __init__.py
│   ├── data.py            # Data loading, cleaning, discretization, and literal encoding
│   ├── sequential_covering.py # Sequential Covering with FOIL gain and Cohen pruning
│   ├── bayesian_rule_list.py  # Bayesian Rule List with Dirichlet likelihood and MCMC
│   ├── evaluation.py      # 60/20/20 benchmark protocol and validation threshold tuning
│   └── visualization.py   # Decision lists, ROC, calibration, and confusion matrix plots
├── tests/
│   ├── __init__.py
│   └── test_smoke.py      # Automated smoke and sanity tests
└── results/
    ├── evaluation_raw.csv
    ├── evaluation_pareto.png
    └── figures/
        ├── sequential_covering_decision_list.png
        ├── bayesian_rule_list_decision_list.png
        ├── roc_curves.png
        ├── confusion_matrices.png
        └── calibration.png
```

---

## Running Tests

To verify that the preprocessing pipeline and both model implementations function correctly, run the automated test suite:

```bash
python -m pytest tests/
```

Full technical details, mathematical formulations, implementation challenges, solutions, and member contributions are documented in [report.md](report.md).
