# Interpretable Rule Learning in PyTorch

This is a shared four-member group project. It implements the two assigned topics—**Sequential Covering** and **Bayesian Rule Lists**—from scratch using PyTorch as the core implementation framework. The models predict whether an Adult Census record has annual income greater than `$50K`.

> Working notes, the assignment prompt, and the presentation plan live in `.dev/` (shared, gitignored) and are not part of the submitted deliverable.

## Project contents

- `adult_census_rule_models.py` — preprocessing, both classifiers, training, and a single-run evaluation.
- `member4_evaluation_audit.py` — multi-seed evaluation, validation threshold tuning, complexity sweeps, CSV summaries, and a Pareto plot.
- `test_preprocessing.py` — preprocessing inspection script.
- `adult.data` — UCI Adult training data used by the default experiment.
- `adult.test` — original UCI test file, retained with the dataset files.
- `report.md` — project report.
- `presentation_outline.md` — presentation plan.

## Requirements

Use **Python 3.12**. PyTorch may not provide a compatible wheel for Python 3.14.

```bash
python3.12 -m venv .venv
source .venv/bin/activate                 # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On this machine, the Python 3.12 executable is `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.

## Run the project

Run preprocessing inspection:

```bash
.venv/bin/python test_preprocessing.py
```

Run one stratified 80/20 experiment for both models:

```bash
.venv/bin/python adult_census_rule_models.py
```

Run the multi-seed audit. The default is three seeds and 4,000 BRL search iterations per fit:

```bash
.venv/bin/python member4_evaluation_audit.py --data adult.data --seeds 3 --brl_iter 4000 --out member4_results
```

The audit writes `member4_results_raw.csv`, `member4_results_table.csv`, and `member4_results_pareto.png`.

## Implementation notes

The algorithms are not imported from a rule-learning package:

- Sequential Covering grows rules using FOIL gain, prunes them using Cohen's metric, removes covered examples, and assigns a residual default probability.
- Bayesian Rule Lists mine frequent one- and two-literal candidates, score lists using a Beta-binomial leaf likelihood and a Poisson-style list-length prior, and search with add/remove/swap/replace simulated-annealing moves.
- PyTorch tensors represent literal coverage masks and perform the main count and scoring operations.

Pandas, NumPy, and scikit-learn are used only for data handling, array conversion, splitting, and evaluation metrics. Matplotlib is used only for the optional Pareto plot.
