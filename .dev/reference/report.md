# Interpretable Rule Learning with PyTorch

## 1. Project overview

This project implements and compares two interpretable classification methods:

1. **Sequential Covering**, implemented as a RIPPER-style separate-and-conquer learner.
2. **Bayesian Rule Lists (BRL)**, implemented as a probabilistic ordered list of rules searched with an annealed MAP procedure.

The implementation is written for the project rather than imported from an open-source rule-learning package. PyTorch is used for the core tensor representation and calculations. The goal is to predict whether an individual earns more than $50,000 per year while keeping every prediction traceable to human-readable IF–THEN conditions.

The code is in `/Users/yashwantkotikala/Projects/IML/adult_census_rule_models.py`. The accompanying README contains installation and execution instructions.

## 2. Dataset

We use the **UCI Adult Census Income** dataset. It contains demographic, education, employment, and financial attributes collected from US Census data. The target is binary: `<=50K` or `>50K`.

The supplied `adult.data` file contains 32,561 rows and 14 input columns plus the target. Some rows contain `?` values. After treating these as missing values and dropping incomplete rows, 30,162 observations remain. The class proportions are approximately 75.1% for `<=50K` and 24.9% for `>50K`.

This dataset is appropriate because its feature names have a natural semantic meaning and its binary target is suitable for rule classification. A rule involving education, occupation, marital status, or hours worked can be inspected without translating a numerical neural-network representation. The dataset also contains enough observations for a stratified train/test split.

### Preprocessing

`fnlwgt` is removed because it is a survey sampling weight rather than an interpretable personal characteristic. `education-num` is removed because textual `education` is more readable. Continuous variables are discretised into domain-motivated categories:

- age: Young (<=25), Middle-aged (26–45), Senior-Adult (46–65), Senior (>65)
- hours per week: Part-time (<35), Full-time (35–40), Overtime (41–55), Extreme (>55)
- capital gain: None, Moderate (<7k), High (>7k)
- capital loss: None, Low-Loss (<=1.8k), High-Loss (>1.8k)

`native-country` is simplified to `United-States` versus `Other` to reduce sparse literals. The target is encoded as 0/1. The default experiment uses an 80/20 stratified split.

## 3. Libraries

- **PyTorch:** stores literal coverage as Boolean tensors and performs vectorised FOIL-gain, support, pair-count, and leaf calculations.
- **Pandas:** reads the Adult CSV, handles missing values, strips text, and performs discretisation.
- **NumPy:** converts cleaned labels and arrays into formats consumed by PyTorch and evaluation code.
- **scikit-learn:** supplies stratified splitting and standard metrics only; it does not implement either rule learner.
- **Matplotlib:** creates the optional F1-versus-complexity Pareto plot.
- **Python standard library:** supplies random-number control, logarithms, gamma functions, timing, and argument parsing.

The environment was tested with Python 3.12, PyTorch 2.15.0, pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1, and Matplotlib 3.11.2. Exact versions are in `requirements.txt`.

## 4. Algorithm 1: Sequential Covering

Sequential Covering follows separate-and-conquer. The learner repeatedly constructs a rule for positive examples, adds it to the list, removes examples it covers, and continues on remaining data. A default rule handles uncovered examples.

Each condition such as `[education = Bachelors]` is a literal. The implementation creates one Boolean PyTorch row per literal. The coverage tensor `M` has shape `(number of literals, number of examples)`; multi-condition coverage is the element-wise AND of selected rows.

Starting with an empty rule, every unused literal is evaluated. For each candidate, the learner counts covered positives and negatives and calculates FOIL gain:

`gain = p1 * (log2(precision_after) - log2(precision_before))`.

The greatest positive-gain literal is added. Growth stops at the maximum condition count or when no candidate improves precision.

The grown rule is evaluated on a held-out portion of alive examples. Trailing conditions are removed when doing so improves Cohen's metric `(p - n) / (p + n)`. Rules that do not cover more positives than negatives are rejected. The accepted rule is removed from the alive set, and the process repeats. Rule probabilities are observed precision, optionally Laplace-smoothed; the default probability is the residual positive rate.

## 5. Algorithm 2: Bayesian Rule Lists

A Bayesian Rule List is ordered. The first rule covering a record determines its leaf probability; otherwise the default leaf is used. Later rules are therefore evaluated only after earlier rules fail.

The implementation mines frequent one-literal candidates and frequent compatible two-literal candidates. Pair support is counted using a PyTorch matrix multiplication of Boolean coverage rows. Pairs containing two values from the same column are excluded.

For each leaf, the code uses a Beta-binomial marginal likelihood. With `n0` negatives and `n1` positives, the log leaf score uses gamma functions and Beta-binomial conjugacy. A Poisson-style prior rewards short lists, while a cardinality penalty discourages two-condition rules:

`score = log likelihood + log list-length prior - width penalty`.

## 6. Results

The following results were computed by running `adult_census_rule_models.py` with seed 42 and an 80/20 stratified split. They are not hardcoded into the program.

| Model | Accuracy | Precision | Recall | F1 | AUC | Fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Sequential Covering | 0.8263 | 0.6967 | 0.5353 | 0.6054 | 0.7354 | 0.1188 |
| Bayesian Rule List | 0.8319 | 0.7526 | 0.4840 | 0.5891 | 0.8619 | 1.3682 |

The Sequential Covering confusion matrix was TN=4,181, FP=350, FN=698, TP=804. The BRL confusion matrix was TN=4,292, FP=239, FN=775, TP=727.

Sequential Covering obtains higher recall and slightly higher F1 in this run, useful when finding more high-income examples matters. BRL obtains higher accuracy, precision, and especially AUC, indicating stronger ranking/probability separation and fewer false positives at threshold 0.5. BRL is slower because it searches many candidate lists, while Sequential Covering is a fast greedy learner.

The `member4_evaluation_audit.py` script provides a stronger protocol: it separates training, validation, and test data, tunes the classification threshold only on validation data, repeats configurations across seeds, and records explanation size. It also explores rule-count, rule-width, list-prior, and minimum-support settings.

## 7. Challenges and solutions

### Challenge 1: Continuous features do not directly form readable literals

Rules require propositions such as `age_group = Middle-aged`, not arbitrary real-valued thresholds. We used domain-informed bins for age, hours, gain, and loss. This preserves interpretability and avoids many nearly unique thresholds.

### Challenge 2: Class imbalance

Only about one quarter of cleaned records are above $50K. Accuracy alone would overstate quality. We report precision, recall, F1, AUC, and confusion matrices, and the audit includes validation-only threshold tuning for the minority class.

### Challenge 3: BRL has a large combinatorial search space

We restrict candidates to frequent singletons and pairs, use a simplicity prior, cap list length, and use simulated annealing with reproducible seeds. This makes the experiment tractable while retaining add, remove, swap, and replace moves.

### Challenge 4: Maintaining a from-scratch core implementation

No external RIPPER or BRL learner is used. Coverage, FOIL gain, pruning, candidate support, Beta-binomial scoring, posterior probabilities, and list search are implemented in the project. Supporting packages are limited to allowed data, array, split, metric, and plotting tasks.

### Challenge 5: Reproducibility and fair evaluation

Random seeds are passed to PyTorch and Python's random generator where appropriate. The audit fits models only on training data, chooses optional thresholds on validation data, and evaluates on an untouched test subset. This avoids using test performance to select a model.

## 8. Limitations and responsible use

The Adult dataset is historical and contains sensitive attributes such as race and sex. High predictive performance must not be interpreted as evidence that these attributes should be used in real employment, credit, or public-benefit decisions. The rules reveal associations, not causal relationships. These are educational implementations and should not be deployed for consequential decisions without fairness, privacy, legal, and domain review.

The current BRL search is a MAP-style heuristic rather than an exact posterior sampler. Candidate rules are limited to one or two literals, and discretisation choices influence discovered rules. Performance is an empirical comparison, not a claim that settings are globally optimal.

## 9. Group member roles and contributions

This report, the source code, and the presentation are a shared product of a
four-member group. Ownership is divided by workstream, while the dataset,
evaluation protocol, and final write-up were reviewed by every member.

| Workstream | Owner | Key contributions |
|---|---|---|
| Dataset, preprocessing, problem formulation | Member 1 | Obtained and described the Adult dataset; handled missing values; designed interpretable discretisation (`age`, `hours`, `capital gain/loss`, `native-country`); encoded the target; documented the pipeline. |
| Sequential Covering (RIPPER-style) | Member 2 | Literal coverage representation; FOIL-gain rule growth; Cohen-metric pruning; separate-and-conquer loop; default rule; rule printing and sanity checks. |
| Bayesian Rule Lists | Member 3 | Candidate mining (singletons + compatible pairs); pair support via PyTorch matmul; Beta-binomial leaf likelihood; list-length and cardinality priors; add/remove/swap/replace annealed search; posterior leaf probabilities. |
| Evaluation, audit, integration | Member 4 | Train/validation/test protocol; validation-only threshold tuning; multi-seed runs; complexity sweeps; Pareto analysis; metric reporting; README and reproducibility checks. |
| Shared (all members) | All | Final code review; report review; presentation; metric verification against program output. |

Names can be filled in by the group in place of the `Member 1`–`Member 4`
labels if the professor prefers real names on the report.

## 10. Conclusion

Both models produce compact, auditable decision logic on the Adult dataset. Sequential Covering is faster and finds more positive cases at the default threshold in the reported run. Bayesian Rule Lists provide ordered probabilistic leaves, stronger precision and AUC, and an explicit simplicity prior, at the cost of additional search time. Together they demonstrate the interpretable-ML trade-off between greedy efficiency, probabilistic calibration, predictive behavior, and explanation complexity.


The starting list is the best individual candidate. Simulated annealing proposes add, remove, swap, and replace moves. Worse proposals can be accepted early in the search, while the temperature decreases from 1.0 toward 0.2. The best list encountered is retained. This is an annealed MAP-style search, not exact MCMC posterior sampling.

Each leaf reports a smoothed posterior mean:

`P(>50K | leaf) = (n1 + alpha1) / (n0 + n1 + alpha0 + alpha1)`.
