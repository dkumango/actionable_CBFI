"""
E0. Corrected re-run of the benchmark with end-to-end evaluation counting and a LEWIS-style
baseline.

Protocol: real_data.load_task (stratified 80:20 split, train-only fitting, corrected
actionability), adverse TEST queries sampled at random (up to N per dataset x model x seed).

Methods per query:
  acbfi_v2        revised A-CBFI (best-response diagnosis + staged single/pair/triple search)
  acbfi_v1        manuscript A-CBFI (mean-based diagnosis, 25th-pct pruning, IGBS)
  untargeted      manuscript "Untargeted Causal" (beam search over all mutable features)
  shap_staged     ablation: acbfi_v2's search driven by |SHAP| instead of best-response C_G1
  lewis_style     LEWIS-style recourse on the same SCM
  shap_ce         SHAP-targeted CE with real SHAP, non-causal (manuscript semantics)
  random_grid_ce  manuscript "Wachter's CE" (renamed), non-causal
Metrics: success, MAD-L1 cost (all features), L0, L_active, evaluations (diagnosis /
search / total), time, stage, cross-model validity, data-tail use.

Usage: python E0_benchmark.py <dataset|all> [n_queries] [seed]
Output: exp_result/E0/E0_<dataset>_seed<seed>.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../acbfi", "../real", "../synthetic"):
    sys.path.insert(0, os.path.join(HERE, p))
from baselines import lewis_style, random_grid_ce, shap_targeted_ce  # noqa: E402
from core import (CountingModel, acbfi_v1, acbfi_v2, acbfi_v3, mad_of,  # noqa: E402
                  staged_by_score, untargeted)
from R1_lever_vs_cause import make_models  # noqa: E402
from real_data import SPECS, load_task  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "exp_result", os.environ.get("E0_OUT", "E0"))
os.makedirs(OUT, exist_ok=True)
CAUSAL = {"acbfi_v3", "acbfi_v2", "acbfi_v1", "untargeted", "shap_staged", "lewis_style"}


def run(dataset, n_q=100, seed=0):
    out_csv = os.path.join(OUT, f"E0_{dataset}_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"skip {out_csv}"); return
    t_start = time.time()
    task = load_task(dataset, seed=seed)
    act = task.actionable
    immutable = [i for i in range(len(task.features)) if i not in act]
    rng = np.random.default_rng(seed)
    bg = task.X_train[rng.choice(len(task.X_train), min(2000, len(task.X_train)), replace=False)]
    mad = mad_of(bg)
    lo, hi = np.quantile(task.X_train, 0.025, axis=0), np.quantile(task.X_train, 0.975, axis=0)
    fns = {}
    for name, clf in make_models().items():
        clf.fit(task.X_train, task.y_train)
        fns[name] = (lambda X, c=clf: c.predict_proba(np.asarray(X))[:, 1])
    rows = []
    for mname, fn in fns.items():
        cand = np.where(fn(task.X_test) > 0.5)[0]
        if len(cand) == 0:
            print(f"{dataset} {mname}: no adverse test queries"); continue
        others = [f for n, f in fns.items() if n != mname]
        shap_counter = CountingModel(fn)
        explainer = shap.explainers.Permutation(shap_counter, shap.maskers.Independent(bg[:100]))
        model = CountingModel(fn)
        for q, i in enumerate(rng.choice(cand, size=min(n_q, len(cand)), replace=False)):
            x = task.X_test[i]
            shap_counter.reset()
            t0 = time.time()
            phi = explainer(x[None, :], max_evals=max(500, 2 * len(x) + 1), silent=True).values[0]
            shap_time, shap_evals = time.time() - t0, sum(shap_counter.counts.values())
            qrng = np.random.default_rng(seed * 100000 + q)
            results = {
                "acbfi_v2": acbfi_v2(model, task.scm, x, bg, act, rng=qrng),
                "acbfi_v1": acbfi_v1(model, task.scm, x, bg, act, rng=qrng),
                "untargeted": untargeted(model, task.scm, x, bg, act),
                "shap_staged": staged_by_score(model, task.scm, x, bg, act,
                                               {j: abs(phi[j]) for j in act}, diag_evals=shap_evals),
                "lewis_style": lewis_style(model, task.scm, x, bg, act, immutable),
                "shap_ce": shap_targeted_ce(model, x, bg, act, phi, shap_evals=shap_evals,
                                            scm=task.scm),
                "random_grid_ce": random_grid_ce(model, x, bg, act, scm=task.scm),
                "acbfi_v3": acbfi_v3(model, task.scm, x, bg, act,
                                     rng=np.random.default_rng(seed * 100000 + q)),
            }
            for meth, r in results.items():
                rec = {"dataset": dataset, "seed": seed, "model": mname, "query": q,
                       "test_row": int(i), "method": meth, "success": r.success,
                       "evals_diag": r.evals_diag, "evals_search": r.evals_search,
                       "evals_total": r.evals_diag + r.evals_search,
                       "time": r.time + (shap_time if meth in ("shap_staged", "shap_ce") else 0.0),
                       "stage": r.stage, "cost": np.nan, "cost_action": np.nan,
                       "L0": np.nan, "L_active": np.nan,
                       "cross_valid": np.nan, "tail": np.nan, "intervened": "", "changed": ""}
                if r.success:
                    ch = np.abs(r.x_cf - x) > 1e-6
                    dev = np.abs(r.x_cf - x) / mad
                    rec["cost"] = float(dev.sum())
                    rec["cost_action"] = float(dev[r.intervened].sum()) if meth in CAUSAL \
                        else float(dev.sum())
                    rec["L0"] = int(ch.sum())
                    rec["L_active"] = len(r.intervened) if meth in CAUSAL else int(ch.sum())
                    rec["cross_valid"] = float(np.mean([f(r.x_cf[None])[0] < 0.5 for f in others]))
                    rec["tail"] = bool(np.any(ch & ((r.x_cf < lo) | (r.x_cf > hi))))
                    rec["intervened"] = str([task.features[j] for j in r.intervened])
                    rec["changed"] = str([task.features[j] for j in np.where(ch)[0]])
                rows.append(rec)
        print(f"{dataset} seed{seed} {mname} done ({time.time() - t_start:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    paper = ["heloc", "german_credit", "taiwan_default", "gmsc", "adult", "financial_loan",
             "cdc_diabetes", "diabetes", "student_dropout", "student_perf"]   # the ten datasets of the paper
    for d in (paper if ds == "all" else [ds]):
        run(d, n_q, seed)
