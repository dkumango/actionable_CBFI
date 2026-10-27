# Method ↔ code map

This file maps each method element of the manuscript to the code that implements it.

**When to use it:** check every row against the code before each submission build. If code and text disagree, change the text, or rerun the affected experiments.

Last checked: 2026-10-27, against `exp_code/acbfi/core.py`, `exp_code/synthetic/synth_lib.py` and `exp_code/experiments/*.py`.

| Manuscript | What it says | Code | Parameters in code |
|---|---|---|---|
| 3.1, eq. (1) cost | MAD-scaled L1 over all features, incl. downstream | `core._search_targets`, `core.acbfi_v3` (`cost = (|Xcf - x| / mad).sum`) | `mad_of(bg)`: MAD on 2,000 background training rows; MAD < 1e-6 set to 1 |
| 3.1 queries | adverse if h(x) > τ; queries with h(x) > 0.5 | `experiments/E0_benchmark.py` (`fn(X_test) > 0.5`) | up to 100 per model × seed |
| 3.1 success | h(x^cf) < τ | `core.THRESHOLD` (0.5); E7 varies τ | |
| 3.1 constraints | direction and bounds applied to every do() | `real_data.ConstrainedSCM` (`allowed`, clipping in `do`) | |
| 3.2, eq. (2) IGBS averages | mean over background rows; pair values from the same row | `synth_lib.diagnose` (`rows = background[...]`, same `rows` for singles and pairs) | 50 rows |
| 3.2, eq. (3) F1 | max over grid V_i of the reduction | `core.acbfi_v3` Step 1 (`grid_v1`, `F1[i] = max(0, max(p0 - p))`) | 6 quantile levels (min…max), filtered by direction/bounds; clipped at 0 |
| 3.2/3.4 F2 (pair ranking) | best joint reduction over 30 background rows; pairs ranked by F2 = F1+F1+C_G4^br | `core._best_response_pairs`, `core.acbfi_v3` (`pair_score`) | `n_pair_samples = 30`; clipped at 0 |
| 3.2, eq. (4) PG1/PG3/PG4 | frequencies over background rows, same rows for singles and pairs | `experiments/E8_cg3_cg4.flip_matrix`, `pg` | 30 rows |
| 3.4 Step 2 | cheapest successful single action over all actionable features | `core.acbfi_v3` (`best` over all `act`) | |
| 3.4 Step 3 | top-M by F1; top-P pairs by F2; then top-T triples (best pair + best third by F2) | `core.acbfi_v3` | M = 6, P = 3, T = 2 |
| 3.4 A-CBFI-topM | all pairs, then all triples, of top-M | `core.acbfi_v3(escalate="topM")` | |
| 4.3 IGBS | |C_G1|+|C_G4| pruning below 25th percentile, at most 15 features, beam search | `core.acbfi_v1`, `core.igbs_beam` | 50 rows; width 5; depth ≤ 5; budget 500 |
| 4.3 untargeted | same beam search over all actionable features, no priority | `core.untargeted` | budget 500 |
| 4.3 LEWIS-style | sufficiency over context neighbours ≥ 0.5; singles then pairs | `acbfi/baselines.lewis_style` | |
| 4.3 SHAP-staged | staged search ranked by |SHAP|; pair score = sum of |SHAP| | `core.staged_by_score` | 100 background rows |
| 4.3 SHAP-CE, DiCE | non-causal | `shap_targeted_ce` (called in `experiments/E0_benchmark.py`), `experiments/E5_dice.py` | dice-ml 0.12 random |
| 5.3 Table 4 rules | `br_cg4` = F2 on 30 rows; `mean_cg4` = IGBS average; `additive` = F1+F1 | `experiments/E6_escalation_ablation.py` | |
| 3.3 closed forms | signs of Lemma 1 (b), (c), Prop. 1, App. B | `manuscript/check_formulas.py` (sympy) | |
| Comparative claims | e.g. "never significantly worse", "on every dataset" | `make_assets.py` verdicts (`main_total.sig_worse_*`, `*_all` flags in assets.json) | |
