# Actionable CBFI

**Prescriptive Interventional Diagnosis and Interaction-Aware Escalation for Sparse Causal Algorithmic Recourse**

This repository contains the code and results of the revised A-CBFI paper.

A-CBFI computes causal recourse on a structural causal model (SCM) in three steps:
1. It scores each actionable feature by the best probability change that a do-intervention can reach (`F1`).
2. It prescribes the cheapest single action when one exists.
3. It searches feature pairs and then triples only when no single action is enough. Pairs are ranked by their best joint gain, `F2 = F1(i) + F1(j) + C_G4^br`.

The output separates the direct actions from the downstream changes that the SCM predicts. Because Step 1 evaluates every actionable feature, A-CBFI also returns the menu of substitutable single actions at no extra cost.

> **Preliminary version.** An earlier version of this work is arXiv:2608.27821v1. In the paper its method is called **IGBS** (averaged interventional scores with an interaction-guided beam search). The code here reproduces the revised paper, which includes IGBS only as a baseline.

## Repository layout

```
exp_code/
  example.py    minimal A-CBFI example (see "Quick example")
  acbfi/        core.py (A-CBFI, IGBS, untargeted search, staged search), baselines.py (LEWIS-style, SHAP-CE)
  real/         real_data.py (datasets, actionability, constraints, DAGs, SCM), R1_lever_vs_cause.py (model definitions)
  synthetic/    synth_lib.py (SCM, interventional diagnosis, exact SHAP), S1_*.py, S2_*.py (Section 5.1)
  experiments/  E0_benchmark.py … E8_cg3_cg4.py (Sections 5.2–5.7, Appendix C)
  manuscript/   make_assets.py (all tables, figures and numbers of the paper), make_fig1_pptx.py,
                check_formulas.py (sympy check of the closed forms), method_code_map.md
exp_result/
  E0m/ E4/ E5/ E6/ E7/ E8/ synthetic/   raw per-query results (CSV) and summaries
  manuscript_assets/                     assets.json (every number in the paper), Fig. 1–4
data/original/                           four datasets of the preliminary version (included)
data/<name>/                             the other six datasets (download, see below)
```

The repository contains only what the paper uses. Run logs, superseded experiments, and the scripts for exploratory analyses not reported in the paper are left out.

## Requirements

Tested with Python 3.14 and the following packages:

| package | version | used for |
|---|---|---|
| numpy, pandas, scipy | 2.4.3, 3.0.1, 1.17.1 | all |
| scikit-learn, xgboost | 1.8.0, 3.2.0 | random forest, RBF-SVM, XGBoost |
| shap | 0.51.0 | SHAP baselines |
| dice-ml | 0.12 | DiCE baseline (E5) |
| networkx, causal-learn | 3.6.1, 0.1.4.8 | DAG variants and PC algorithm (E4) |
| matplotlib, python-pptx | 3.10.8, 1.0.2 | figures |
| sympy | 1.14.0 | `check_formulas.py` |

```bash
pip install numpy pandas scipy scikit-learn xgboost shap dice-ml networkx causal-learn matplotlib python-pptx sympy
```

## Data

`exp_code/real/real_data.py` reads the datasets from the paths below.

**Included.** The four datasets of the preliminary version are in `data/original/`. They are the same files, with the same preprocessed column names, as in the earlier release.

| Dataset | File | Source | Adverse outcome |
|---|---|---|---|
| German Credit | `data/original/german_credit.csv` | UCI Statlog German Credit | default = 2 (bad) |
| Adult Income | `data/original/adult.csv` | UCI Adult | income ≤ 50K |
| Lending Loan | `data/original/financial_loan.csv` | Kaggle "Financial Loan" (Lending Club subset) | Charged Off |
| Pima Diabetes | `data/original/diabetes.csv` | UCI / Kaggle Pima Indians Diabetes | Outcome = 1 |

**Not included.** Download the other six datasets from their sources and save them under the file names below.

| Dataset | Expected file | Source | Adverse outcome |
|---|---|---|---|
| HELOC | `data/heloc/FICO-HELOC-cleaned.arff` | OpenML dataset 45553, https://www.openml.org/d/45553 (FICO Explainable Machine Learning Challenge; FICO terms apply) | RiskPerformance = Bad |
| Taiwan Default | `data/taiwan/taiwan_default.csv` | UCI dataset 350, Default of Credit Card Clients, https://archive.ics.uci.edu/dataset/350 (columns `X1`–`X23`, `Y`) | default next month |
| Give Me Some Credit | `data/gmsc/GiveMeSomeCredit.arff` | OpenML dataset 46929, https://www.openml.org/d/46929 | distress within 2 years |
| CDC Diabetes | `data/cdc/cdc_diabetes.csv` | UCI dataset 891, CDC Diabetes Health Indicators, https://archive.ics.uci.edu/static/public/891/data.csv | Diabetes_binary = 1 |
| Student Dropout | `data/student_dropout/student_dropout.csv` | UCI dataset 697, Predict Students' Dropout and Academic Success, https://archive.ics.uci.edu/dataset/697 | Target = Dropout |
| Student Performance | `data/student_perf/student_perf.csv` | UCI dataset 320, Student Performance, https://archive.ics.uci.edu/dataset/320 (Portuguese course, `student-por.csv`, saved comma-separated) | G3 < 10 |

Appendix A of the paper lists, for each dataset, the actionable features, the direction and bound constraints, and the DAG. All of these are defined in `real_data.py` (`SPECS`, `DIRECTIONS`, `BOUNDS`).

## Quick example

`exp_code/example.py` runs A-CBFI on one query of the Pima Diabetes data, which is included in `data/original/`:

```bash
cd exp_code
python example.py
```

The core of the example is:

```python
from core import CountingModel, acbfi_v3            # exp_code/acbfi
from R1_lever_vs_cause import make_models           # exp_code/real
from real_data import load_task                     # exp_code/real

task = load_task("diabetes", seed=0)                # data, actionable set, constraints, SCM
clf = make_models()["RF"].fit(task.X_train, task.y_train)
h = CountingModel(lambda X: clf.predict_proba(X)[:, 1])     # adverse probability, counts evaluations
bg = task.X_train[:2000]                            # background rows
x = task.X_test[4]                                  # a query with h(x) > 0.5

res = acbfi_v3(h, task.scm, x, bg, task.actionable)
res.stage            # 'single', 'pair' or 'triple'
res.intervened       # indices of the direct actions
res.x_cf             # counterfactual after SCM propagation (standardised units)
res.extra["F1"]      # Step 1 diagnosis: best reachable probability reduction per feature
```

Output (random forest, seed 0):

```
query: test row 4, adverse probability h(x) = 0.55
F1 (best reachable probability reduction): glucose 0.30, bmi 0.20, bloodpressure 0.08
stage: single, model evaluations: 28, h(x_cf) = 0.46
direct actions:    glucose 151.0 -> 125.0
downstream (SCM):  insulin 120.0 -> 93.1
menu of single actions (cost order):
  glucose 151.0 -> 125.0   cost 1.54, h = 0.46
  bmi 35.5 -> 30.0   cost 2.15, h = 0.48
  bloodpressure 62.0 -> 122.0   cost 7.50, h = 0.47
```

A single direct action (lower glucose) is enough here. The SCM predicts the downstream change in insulin, which is reported separately from the direct action. The menu lists the other single actions that also reach the decision boundary.

The paper uses no direction constraints for Pima, so the menu can contain implausible actions such as raising blood pressure. Direction and bound constraints are defined per dataset in `DIRECTIONS` and `BOUNDS` in `real_data.py`. They are applied to every intervention.

## Reproducing the paper

Run the commands from `exp_code/`, in this order. The commands are written for bash (Linux, macOS, Git Bash). Each experiment writes per-seed CSV files to `exp_result/<experiment>/`. Experiments E0–E8 skip a dataset/seed whose output file already exists, so delete that file to recompute it. The synthetic scripts S1 and S2 always overwrite their output.

```bash
cd exp_code
DS="heloc german_credit taiwan_default gmsc adult financial_loan cdc_diabetes diabetes student_dropout student_perf"

# Section 5.1 - synthetic validation (Lemma 1, Proposition 1, Fig. 2)
python synthetic/S1_interaction_sources.py
python synthetic/S2_planted_synergy.py 40 lever
python synthetic/S2_planted_synergy.py 40 cause

# Section 5.2 - main benchmark (Table 3): 10 datasets x 3 seeds x 3 models, up to 100 queries per model
for s in 0 1 2; do for d in $DS; do E0_OUT=E0m python experiments/E0_benchmark.py $d 100 $s; done; done
python experiments/E0m_summarize.py            # McNemar / Wilcoxon-Holm summaries
for s in 0 1 2; do python experiments/E5_dice.py all $s; done        # DiCE column (six datasets)

# Section 5.3 - escalation-stage ablation (Table 4)
for s in 0 1 2; do python experiments/E6_escalation_ablation.py all $s; done

# Section 5.4 - difficulty sweep (Fig. 3)
for s in 0 1 2; do python experiments/E7_difficulty_sweep.py all $s; done

# Section 5.5 - sensitivity to the causal graph (Fig. 4), seed 0
python experiments/E4_dag_sensitivity.py all 0

# Section 3.2, Section 5.7, Appendix C - CBFI groups G3/G4, substitutable single actions (Table 5)
python experiments/E8_cg3_cg4.py synth
for s in 0 1 2; do python experiments/E8_cg3_cg4.py all $s; done
python experiments/E8_cg3_cg4.py summary

# Tables, figures and every number of the paper -> exp_result/manuscript_assets/assets.json
python manuscript/make_assets.py
python manuscript/make_fig1_pptx.py           # Fig. 1 (editable PowerPoint; export to PNG)
python manuscript/check_formulas.py           # symbolic check of the signs in Section 3.3 / Appendix B
```

The E4–E8 experiments reuse the queries (test rows, models, seeds) of `exp_result/E0m`, so E0 must be run first.

Running `make_assets.py` on the results in this repository reproduces the `assets.json` used for the paper exactly.

## Where each result comes from

| Paper | Script | Result files |
|---|---|---|
| Section 5.1, Fig. 2 | `synthetic/S1_interaction_sources.py`, `synthetic/S2_planted_synergy.py` | `exp_result/synthetic/S1_*.csv`, `S2_*.csv` |
| Table 2 | `real/real_data.py` via `manuscript/make_assets.py` | `assets.json` (`datasets`) |
| Table 3, Section 5.2 | `experiments/E0_benchmark.py`, `E0m_summarize.py`, `E5_dice.py` | `exp_result/E0m/`, `exp_result/E5/` |
| Table 4, Section 5.3 | `experiments/E6_escalation_ablation.py` | `exp_result/E6/` |
| Fig. 3, Section 5.4 | `experiments/E7_difficulty_sweep.py` | `exp_result/E7/` |
| Fig. 4, Section 5.5 | `experiments/E4_dag_sensitivity.py` | `exp_result/E4/` (incl. the learned/perturbed edge lists) |
| Section 5.6 (case studies) | `manuscript/make_assets.py` (`case_studies`) | `assets.json` (`cases`) |
| Section 3.2 (OR/AND), Table 5, Section 5.7, Appendix C | `experiments/E8_cg3_cg4.py` | `exp_result/E8/` |
| Fig. 1 | `manuscript/make_fig1_pptx.py` | `exp_result/manuscript_assets/fig1_pipeline.*` |

`exp_code/manuscript/method_code_map.md` maps each equation and algorithm step of the paper to the function and parameters that implement it.

### Method names in the result files

| In the files | In the paper |
|---|---|
| `acbfi_v3` | A-CBFI (F2-screened escalation) |
| `v3_topM` (E7), `topM_all` (E6/E8) | A-CBFI-topM (exhaustive escalation over the top-M levers) |
| `v3_single` (E7) | Steps 1–2 only (no escalation) |
| `acbfi_v1` | IGBS (preliminary version) |
| `untargeted` | untargeted causal beam search |
| `lewis_style` | LEWIS-style causal recourse |
| `shap_staged` | SHAP-staged search |
| `shap_ce` | SHAP-CE (non-causal) |
| `dice_random` (E5) | DiCE (non-causal) |
| `br_cg4` (E6/E8) | best-response joint gain F2 (the A-CBFI pair rule) |
| `acbfi_v2`, `random_grid_ce` (E0m) | intermediate variants that are not reported in the paper |

Main columns of the per-query files:
- `success`: whether the method found a counterfactual with h(x^cf) < τ.
- `evals_total`: number of model evaluations, including diagnosis.
- `L_active`: number of direct actions.
- `cost`: MAD-scaled L1 distance, including downstream changes.
- `cost_action`: the same distance restricted to the direct actions.
- `stage`: `single` / `pair` / `triple` / `beam`.
- `intervened`, `changed`: the directly intervened features and all changed features.

## Citation

If you use this code, please cite the paper (under review; the reference will be updated on publication). For the preliminary version, cite:

> Oh S (2026) Actionable CBFI: integrating structural decomposition and causal counterfactual recourse for tabular machine learning. arXiv:2608.27821v1.

Related earlier work by the author:
- Oh S (2022) Predictive case-based feature importance and interaction. *Information Sciences* 593:155–176.
- Oh S (2026) Beyond feature attribution: recovering latent interaction structures in machine learning models via localized CBFI. arXiv:2610.34962.
