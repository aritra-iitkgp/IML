# Final Group Presentation: Interpretable Rule-Based Machine Learning
## Comparative Study of Sequential Covering (RIPPER) vs. Bayesian Rule Lists (BRL) on Adult Census Income

**Course:** Interpretable Machine Learning  
**Target Audience:** Professor, TAs, and Fellow Data Science Peers  
**Total Duration:** 15–20 Minutes (including 3–5 min Q&A)  
**Roles & Speakers:**
- **Member 1:** Data Engineering, Discretization & Problem Formulation
- **Member 2:** Sequential Covering & The RIPPER Algorithm
- **Member 3:** Bayesian Rule Lists (BRL) & Probabilistic Modeling
- **Member 4:** Evaluation Metrics, Trade-offs & Strategic Takeaways

---

## Slide 1: Title & Executive Summary (All Members)
- **Visuals:** Title, team member names and assigned phases, university/course banner, central graphic showing a transparent glass-box decision flowchart vs. an opaque black-box neural net.
- **Key Talking Points:**
  - Introduce the objective: Predicting income levels (>50K vs. <=50K) using inherently interpretable rule models rather than post-hoc explainers.
  - Core question: Can transparent IF-THEN rule models deliver production-grade accuracy while remaining 100% human-auditable?
  - Head-to-head comparison: Greedy separate-and-conquer (RIPPER) vs. Globally optimized Bayesian Rule Lists (BRL).
- **Speaker:** Member 1 (Introduction handoff)

---

## Slide 2: Why Rule Lists? Motivation & High-Stakes Societal Context (Member 1)
- **Visuals:** Infographic of high-stakes domains (Credit underwriting, criminal justice risk scores, government welfare eligibility). A comparison matrix: Inherently Interpretable (Rules, Linear models) vs. Post-Hoc Black Box (SHAP / LIME on XGBoost / Deep Nets).
- **Key Talking Points:**
  - High-stakes decisions demand "Safety by Design" (Cynthia Rudin's thesis: Stop explaining black box models, use interpretable ones instead).
  - The Adult Census dataset mirrors real-world socio-economic lending decisions.
  - Explain why post-hoc explanations can be unfaithful, whereas rule lists offer exact logical paths for every prediction.
- **Speaker:** Member 1 (Duration: 2.5 mins)

---

## Slide 3: Phase 1 — Data Preprocessing & Cognitive Discretization (Member 1)
- **Visuals:** Flowchart of data ingestion (UCI repository / web API), cleaning pipeline, and visual histograms comparing raw continuous distributions vs. binned categories.
- **Key Talking Points:**
  - **Data Cleaning:** Handled missing '?' indicators, dropping 7.4% missing records, preserving 30,162 high-quality observations.
  - **Dropping `fnlwgt`:** Why final weight must be dropped—it is a survey sampling weight, not an interpretable personal attribute. Keeping it generates nonsensical rules.
  - **Discretization Rationale:** Rule learners require discrete propositions.
    - `age` -> 4 demographic stages (Young, Middle-aged, Senior-Adult, Senior).
    - `hours-per-week` -> Part-time, Full-time (standard 40 hrs), Overtime, Extreme.
    - `capital-gain` & `capital-loss` -> Domain-driven bins handling severe zero-inflation (>91% reporting $0).
  - **Data Splitting:** 80/20 stratified split preserving the 24.9% positive class balance.
- **Transition:** "Now Member 2 will demonstrate how we apply Sequential Covering with RIPPER."

---

## Slide 4: Phase 2 — Sequential Covering & The RIPPER Algorithm (Member 2)
- **Visuals:** "Separate-and-Conquer" visual diagram showing:
  1. Full training set ->
  2. Rule 1 covers positive pocket ->
  3. Positive instances removed (Separated) ->
  4. Rule 2 grown on residual data (Conquered) ->
  5. Default rule catches leftovers.
- **Key Talking Points:**
  - **Philosophy:** Rather than dividing the entire space recursively like a Decision Tree, learn one rule at a time for the positive target class (>50K).
  - **Rule Growing:** FOIL's Information Gain greedily chooses literals that maximize purity.
  - **Rule Pruning:** Uses validation prune set and metric $(p - n)/(p + n)$ to prevent overfitting.
  - **Stopping Criterion:** Minimum Description Length (MDL) halts rule growth when rule complexity exceeds error reduction benefits.
- **Speaker:** Member 2 (Duration: 3 mins)

---

## Slide 5: RIPPER Discovered Rules & Linguistic Decomposition (Member 2)
- **Visuals:** Clean syntax box showing the top extracted IF-THEN rules from Wittgenstein's RIPPER:
  - `IF [marital-status=Married-civ-spouse ^ capital_gain_group=High] THEN >50K`
  - `IF [education=Bachelors/Masters ^ occupation=Exec-managerial] THEN >50K`
  - `ELSE <=50K`
- **Key Talking Points:**
  - Walk the audience through the exact decision logic.
  - Explain how the greedy separate-and-conquer order directly impacts which rules appear first.
  - Highlight the simplicity and natural language readability of the learned rules.
- **Transition:** "While RIPPER is fast and greedy, Member 3 will introduce a globally optimized Bayesian alternative."

---

## Slide 6: Phase 3 — Bayesian Rule Lists: Prior vs. Likelihood (Member 3)
- **Visuals:** Bayes' Theorem diagram applied to Rule Lists:
  $$P(\text{Rule List} \mid \text{Data}) \propto P(\text{Data} \mid \text{Rule List}) \times P(\text{Rule List})$$
  along with diagrams of the Poisson prior (penalizing length) and MCMC proposal moves (Add, Swap, Prune, Replace).
- **Key Talking Points:**
  - **The Letham & Rudin (2015) Breakthrough:** Rule learning as Bayesian posterior inference.
  - **The Simplicity Prior (Occam's Razor):** Poisson prior explicitly penalizes long lists and complex rules, preventing bloat mathematically.
  - **Data Likelihood:** Dirichlet-Multinomial conjugate model measures how cleanly each leaf separates income classes.
  - **MCMC Sampling:** Searches the combinatorial space of rule sequences globally without getting trapped in greedy local optima.
- **Speaker:** Member 3 (Duration: 3.5 mins)

---

## Slide 7: The Discovered BRL Decision List & Probabilistic Semantics (Member 3)
- **Visuals:** Visual Decision List card layout:
  - `IF [marital-status=Married-civ-spouse] AND [education=Higher-Ed] THEN P(>50K) = 0.78`
  - `ELSE IF [capital_gain_group=High] THEN P(>50K) = 0.89`
  - `ELSE IF [hours_group=Part-time] THEN P(>50K) = 0.04`
  - `DEFAULT: P(>50K) = 0.12`
- **Key Talking Points:**
  - Highlight key difference: BRL produces *calibrated probabilities* at every step, not just hard 0/1 labels.
  - Show how the ordering enforces fallback logic.
  - Emphasize that the model consists of only 4 concise rules that can be audited by a loan officer in seconds.
- **Transition:** "Member 4 will now present our quantitative head-to-head empirical evaluation."

---

## Slide 8: Phase 4 — Quantitative Performance Evaluation (Member 4)
- **Visuals:** Side-by-side bar chart and performance comparison table:
  - Accuracy, Precision (>50K), Recall (>50K), F1-Score for both models.
  - Confusion matrix heatmaps for RIPPER vs. BRL.
- **Key Talking Points:**
  - **Overall Accuracy:** Both models achieve ~81%–83% accuracy, rivaling complex black-box baselines while retaining 100% transparency.
  - **Precision:** BRL leads in Precision (~70% vs ~65% for RIPPER), meaning false positive rates for high earners are minimized.
  - **Recall:** RIPPER leads in Recall (~58% vs ~52% for BRL), catching more high earners through greedy tail rules.
  - **F1-Score:** Comparable balance (~0.60–0.63).
- **Speaker:** Member 4 (Duration: 3 mins)

---

## Slide 9: Qualitative Trade-Off Analysis: Greedy vs. Bayesian (Member 4)
- **Visuals:** 2x2 Trade-off Matrix comparing:
  - Computational Complexity ($O(N \log N)$ vs. MCMC iterations)
  - Rule Interpretability (Cognitive load, number of conditions)
  - Deterministic Classification vs. Calibrated Posterior Risk
  - Local Optima vulnerability vs. Global MAP convergence
- **Key Talking Points:**
  - Summarize the fundamental machine learning tension: Speed vs. Global Optimality.
  - Cognitive fatigue in Sequential Covering: rule #8 depends on rules #1-#7 having failed.
  - Cleanliness of BRL: Prior guarantees minimal human cognitive load.
- **Speaker:** Member 4

---

## Slide 10: Conclusion, Recommendations & Q&A (All Members)
- **Visuals:** Summary bullet points, GitHub repo link / QR code to source code, open Q&A slide.
- **Key Talking Points:**
  - **When to choose RIPPER:** Large-scale datasets, rapid prototyping, when minority-class recall is critical.
  - **When to choose BRL:** Regulatory compliance, policy and clinical decisions where calibrated probabilities and minimal cognitive load are legally mandatory.
  - Open floor for audience and professor questions.
- **Speaker:** All Members (3–5 mins Q&A)
