"""
R1. How often is the best lever not among the top additive causes on real data?

Protocol (corrected, see real_data.py): stratified 80:20 split, models fitted on train,
N_Q adverse test queries drawn at random, linear-Gaussian SCM fitted on train with the
domain DAG, interventions only on directly actionable features.

For each query and each k (< number of actionable features) every target-selection rule
picks k targets and is searched with the SAME exhaustive grid search under SCM propagation;
the oracle searches all k-subsets. SHAP = model-agnostic permutation SHAP (interventional,
100 background rows) on p_adv.

Lever-not-cause (LNC) flags, among oracle-feasible queries:
  additive_fail : SHAP top-k fails
  all_additive_fail : SHAP, C_G1 and best-response-single top-k all fail
  lnc_recovered : all_additive_fail and the best-response pair rule succeeds

Usage: python R1_lever_vs_cause.py [dataset|all] [n_queries]
Output: exp_result/real/R1_<dataset>_<model>.csv
"""

import os
import sys
import time
from math import comb

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synthetic"))
from real_data import SPECS, load_task  # noqa: E402
from synth_lib import evaluate_instance  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "exp_result", "real")
os.makedirs(OUT, exist_ok=True)

BINS = 5
MAX_ORACLE_ROWS = 200_000   # skip a k whose oracle would need more grid points than this
RULES = ["shap_abs", "cg1", "acbfi_node", "br_single", "br_pair", "random"]


def make_models():
    return {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced",
                                     n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        "XGB": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42),
    }


def run(dataset, n_q, seed=0):
    task = load_task(dataset, seed=seed)
    m = len(task.actionable)
    ks = [k for k in (1, 2, 3) if k < m and comb(m, k) * (BINS + 2) ** k <= MAX_ORACLE_ROWS]
    rng = np.random.default_rng(seed)
    bg = task.X_train[rng.choice(len(task.X_train), min(2000, len(task.X_train)), replace=False)]
    shap_bg = bg[:100]
    for mname, clf in make_models().items():
        out_csv = os.path.join(OUT, f"R1_{dataset}_{mname}.csv")
        if os.path.exists(out_csv):
            print(f"skip {out_csv} (exists)")
            continue
        t0 = time.time()
        clf.fit(task.X_train, task.y_train)
        model = lambda X, c=clf: c.predict_proba(np.asarray(X))[:, 1]
        p_test = model(task.X_test)
        cand = np.where(p_test > 0.5)[0]
        qidx = rng.choice(cand, size=min(n_q, len(cand)), replace=False)
        explainer = shap.explainers.Permutation(model, shap.maskers.Independent(shap_bg))
        rows = []
        for q, i in enumerate(qidx):
            x = task.X_test[i]
            phi = explainer(x[None, :], max_evals=max(500, 2 * len(x) + 1), silent=True).values[0]
            for r in evaluate_instance(model, task.scm, x, bg, task.actionable, ks=ks,
                                       rules=RULES, n_samples=50, rng=rng, phi=phi, bins=BINS):
                r["targets_named"] = str([task.features[int(t)] for t in eval(r["targets"])])
                r["oracle_named"] = str([task.features[int(t)] for t in eval(r["oracle_targets"])]) \
                    if r["oracle_targets"] else ""
                rows.append({"dataset": dataset, "model": mname, "query": q, "test_row": int(i), **r})
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"{dataset} {mname}: {len(qidx)} queries, ks={ks}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_q = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    for d in (list(SPECS) if ds == "all" else [ds]):
        run(d, n_q)
