"""
E7. Difficulty sweep: is the HELOC advantage of A-CBFI v3 a one-dataset artefact, or does
the advantage grow systematically with problem difficulty on EVERY dataset?

Same queries, models, SCMs and constraints as E0m. Difficulty is raised along two axes:
  tau        success requires p_adv < tau, tau in {0.5, 0.4, 0.3, 0.2}
  act_frac   'all' actionable features, or a random 'half' (at least 2) per query
Methods: acbfi_v3 (C_G4-screened escalation), v3_topM (escalation = all pairs then all
triples among the top-M single levers; the E6 winner), v3_single (escalation disabled =
Steps 1-2 only), untargeted (manuscript beam search over the allowed features).

Usage: python E7_difficulty_sweep.py <dataset|all> [seed] [max_queries_per_model]
Output: exp_result/E7/E7_<dataset>_seed<seed>.csv
"""

import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../acbfi", "../real", "../synthetic"):
    sys.path.insert(0, os.path.join(HERE, p))
import core  # noqa: E402
from core import CountingModel, acbfi_v3, mad_of, untargeted  # noqa: E402
from R1_lever_vs_cause import make_models  # noqa: E402
from real_data import load_task  # noqa: E402

E0M = os.path.join(HERE, "..", "..", "exp_result", "E0m")
OUT = os.path.join(HERE, "..", "..", "exp_result", "E7")
os.makedirs(OUT, exist_ok=True)
DATASETS = ["heloc", "adult", "cdc_diabetes", "german_credit", "diabetes", "financial_loan",
            "taiwan_default", "gmsc", "student_dropout", "student_perf"]
TAUS = [0.5, 0.4, 0.3, 0.2]


def run(dataset, seed=0, max_q=100):
    out_csv = os.path.join(OUT, f"E7_{dataset}_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"skip {out_csv}"); return
    t_start = time.time()
    e0 = pd.read_csv(os.path.join(E0M, f"E0_{dataset}_seed{seed}.csv"))
    queries = e0[["model", "query", "test_row"]].drop_duplicates()
    queries = queries[queries["query"] < max_q]
    task = load_task(dataset, seed=seed)
    act = task.actionable
    rng = np.random.default_rng(seed)
    bg = task.X_train[rng.choice(len(task.X_train), min(2000, len(task.X_train)), replace=False)]
    mad = mad_of(bg)
    fns = {}
    for name, clf in make_models().items():
        clf.fit(task.X_train, task.y_train)
        fns[name] = (lambda X, c=clf: c.predict_proba(np.asarray(X))[:, 1])
    rows = []
    try:
        for mname, sub in queries.groupby("model", sort=False):
            model = CountingModel(fns[mname])
            for _, qr in sub.iterrows():
                q, i = int(qr["query"]), int(qr["test_row"])
                x = task.X_test[i]
                srng = np.random.default_rng(seed * 100000 + q + 7)
                half = sorted(srng.choice(act, size=max(2, math.ceil(len(act) / 2)),
                                          replace=False).tolist())
                for frac, A in (("all", act), ("half", half)):
                    for tau in TAUS:
                        core.THRESHOLD = tau
                        for meth in ("acbfi_v3", "v3_topM", "v3_single", "untargeted"):
                            esc = {"acbfi_v3": True, "v3_topM": "topM", "v3_single": False}
                            r = (untargeted(model, task.scm, x, bg, A) if meth == "untargeted"
                                 else acbfi_v3(model, task.scm, x, bg, A,
                                               rng=np.random.default_rng(seed * 100000 + q),
                                               escalate=esc[meth]))
                            rec = {"dataset": dataset, "seed": seed, "model": mname, "query": q,
                                   "act_frac": frac, "tau": tau, "method": meth,
                                   "success": r.success, "stage": r.stage,
                                   "evals": r.evals_diag + r.evals_search,
                                   "cost": np.nan, "L_active": np.nan}
                            if r.success:
                                rec["cost"] = float((np.abs(r.x_cf - x) / mad).sum())
                                rec["L_active"] = len(r.intervened)
                            rows.append(rec)
            print(f"{dataset} seed{seed} {mname} done ({time.time() - t_start:.0f}s)", flush=True)
    finally:
        core.THRESHOLD = 0.5
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def summarize():
    import glob
    from scipy.stats import binomtest
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "E7_*_seed*.csv"))],
                   ignore_index=True)
    pd.set_option("display.width", 250)
    s = df.pivot_table(index=["dataset", "act_frac", "tau"], columns="method", values="success")
    s["gain_vs_untargeted"] = s["acbfi_v3"] - s["untargeted"]
    s["gain_from_escalation"] = s["acbfi_v3"] - s["v3_single"]
    w = df.pivot_table(index=["dataset", "act_frac", "tau", "seed", "model", "query"],
                       columns="method", values="success", aggfunc="first").astype(bool)
    mc = w.groupby(level=["dataset", "act_frac", "tau"]).apply(
        lambda t: pd.Series({"v3_only": int((t.acbfi_v3 & ~t.untargeted).sum()),
                             "unt_only": int((~t.acbfi_v3 & t.untargeted).sum())}))
    mc["p"] = [binomtest(a, a + b).pvalue if a + b else np.nan
               for a, b in zip(mc["v3_only"], mc["unt_only"])]
    s = s.join(mc)
    ev = df.pivot_table(index=["dataset", "act_frac", "tau"], columns="method", values="evals")
    s = s.join(ev.add_prefix("evals_"))
    s.round(4).to_csv(os.path.join(OUT, "E7_summary.csv"), encoding="utf-8-sig")
    print(s.round(3).to_string())
    # dataset-level sign test of the trend: is gain at the hardest tau > gain at tau = 0.5?
    for frac in ("all", "half"):
        g = s.xs(frac, level="act_frac")["gain_vs_untargeted"].unstack("tau")
        inc = (g[min(TAUS)] > g[max(TAUS)]).sum()
        print(f"[{frac}] datasets where v3 gain grows from tau=0.5 to tau={min(TAUS)}: "
              f"{inc}/{len(g)}; mean gain by tau:", g.mean().round(3).to_dict())


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    if ds == "summary":
        summarize(); sys.exit()
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_q = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    for d in (DATASETS if ds == "all" else [ds]):
        run(d, seed, max_q)
