"""
E6. Ablation of the escalation stage (Step 3) of A-CBFI v3 -- the only non-trivial part:
when NO single actionable lever reaches the boundary, which feature pairs should be
searched?

Queries: the E0m queries (same seeds, splits, models, background) on which the single-lever
grid probe (v3 Step 1) finds no recourse. For each such query the Step-1 results (F1 =
best single-lever gain on the grid) are shared, and only the pair-selection rule changes.
Every rule then grid-searches its top-P pairs (P = 3) among the top-M features (M = 6, by
F1), and if none succeeds, its top-T triples (T = 2; best pair + best third by the rule).

Pair rules:
  br_cg4      best-response joint gain on 30 background rows (A-CBFI v3, prescriptive C_G4)
  mean_cg4    manuscript C_G4: mean of [joint gain - single gains] on the same 30 rows
  additive    F1_i + F1_j from Step 1 (no extra evaluations): the "obvious" alternative
  shap        |SHAP_i| + |SHAP_j| (permutation SHAP, its evaluations counted)
  random      random pairs among the top-M
  topM_all    grid-search ALL pairs among the top-M (then all triples among the top-M)
  all_pairs   grid-search ALL actionable pairs (then all triples): exhaustive reference
Metrics: success, MAD-L1 cost, direct actions, evaluations of the escalation stage
(including the rule's own scoring evaluations).

Usage: python E6_escalation_ablation.py <dataset|all> [seed]
Output: exp_result/E6/E6_<dataset>_seed<seed>.csv
"""

import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../acbfi", "../real", "../synthetic"):
    sys.path.insert(0, os.path.join(HERE, p))
from core import THRESHOLD, CountingModel, _search_targets, grid_v1, mad_of  # noqa: E402
from R1_lever_vs_cause import make_models  # noqa: E402
from real_data import load_task  # noqa: E402

E0M = os.path.join(HERE, "..", "..", "exp_result", "E0m")
OUT = os.path.join(HERE, "..", "..", "exp_result", "E6")
os.makedirs(OUT, exist_ok=True)
DATASETS = ["heloc", "adult", "german_credit", "diabetes"]
RULES = ["br_cg4", "mean_cg4", "additive", "shap", "random", "topM_all", "all_pairs"]
BINS, M, P, T, NROWS = 6, 6, 3, 2, 30


def single_probe(model, scm, x, bg, act, mad):
    """v3 Step 1. Returns F1 and the best single-lever recourse (or None)."""
    p0 = model(x[None])[0]
    F1, best = {}, None
    for i in act:
        g = grid_v1(bg, i, BINS, x, scm)
        if len(g) == 0:
            F1[i] = 0.0
            continue
        V = np.zeros((len(g), len(x))); Mk = np.zeros_like(V, bool)
        V[:, i] = g; Mk[:, i] = True
        Xcf = scm.do(x, V, Mk)
        p = model(Xcf)
        F1[i] = max(0.0, float((p0 - p).max()))
        if (p < THRESHOLD).any():
            best = True
    return F1, best


def pair_gains(model, scm, x, rows, pairs, singles):
    """Joint gains per row for each pair, plus per-row single gains -> (best-response, mean C_G4)."""
    p0 = model(x[None])[0]
    br, mean = {}, {}
    for i, j in pairs:
        Mk = np.zeros(rows.shape, bool); Mk[:, [i, j]] = True
        gj = p0 - model(scm.do(x, rows, Mk))
        br[(i, j)] = float(max(0.0, gj.max()))
        mean[(i, j)] = float((gj - singles[i] - singles[j]).mean())
    return br, mean


def search_sets(model, scm, x, bg, mad, sets):
    best = None
    for s in sets:
        r = _search_targets(model, scm, x, list(s), bg, mad, BINS)
        if r and (best is None or r[0] < best[0]):
            best = r
    return best


def escalate(model, scm, x, bg, mad, act, top, score):
    """Top-P pairs by `score`, then top-T triples (best pair + best third)."""
    pairs = sorted(combinations(top, 2), key=lambda p: -score[p])
    best = search_sets(model, scm, x, bg, mad, pairs[:P])
    if best:
        return best, "pair"
    triples = []
    for p in pairs[:T]:
        rest = [t for t in top if t not in p]
        if rest:
            third = max(rest, key=lambda t: max(score.get(tuple(sorted((p[0], t), key=top.index)), 0),
                                                score.get(tuple(sorted((p[1], t), key=top.index)), 0)))
            triples.append(list(p) + [third])
    best = search_sets(model, scm, x, bg, mad, triples)
    return (best, "triple") if best else (None, "")


def run(dataset, seed=0):
    out_csv = os.path.join(OUT, f"E6_{dataset}_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"skip {out_csv}"); return
    t_start = time.time()
    e0 = pd.read_csv(os.path.join(E0M, f"E0_{dataset}_seed{seed}.csv"))
    queries = e0[["model", "query", "test_row"]].drop_duplicates()
    task = load_task(dataset, seed=seed)
    act = task.actionable
    rng = np.random.default_rng(seed)
    bg = task.X_train[rng.choice(len(task.X_train), min(2000, len(task.X_train)), replace=False)]
    mad = mad_of(bg)
    fns = {}
    for name, clf in make_models().items():
        clf.fit(task.X_train, task.y_train)
        fns[name] = (lambda X, c=clf: c.predict_proba(np.asarray(X))[:, 1])
    rows_out = []
    n_hard = 0
    for mname, sub in queries.groupby("model", sort=False):
        model = CountingModel(fns[mname])
        explainer = None
        for _, qr in sub.iterrows():
            q, i = int(qr["query"]), int(qr["test_row"])
            x = task.X_test[i]
            F1, single_ok = single_probe(model, task.scm, x, bg, act, mad)
            if single_ok:
                continue
            n_hard += 1
            qrng = np.random.default_rng(seed * 100000 + q)
            top = sorted(act, key=lambda t: -F1[t])[:M]
            topM_pairs = list(combinations(top, 2))
            rows = bg[qrng.choice(len(bg), NROWS, replace=False)]
            for rule in RULES:
                model.reset(); t0 = time.time()
                if rule in ("br_cg4", "mean_cg4"):
                    p0 = model(x[None])[0]
                    singles = {}
                    for t in top:
                        Mk = np.zeros(rows.shape, bool); Mk[:, t] = True
                        singles[t] = p0 - model(scm_do(task.scm, x, rows, Mk))
                    br, mean = pair_gains(model, task.scm, x, rows, topM_pairs, singles)
                    score = br if rule == "br_cg4" else {k: abs(v) for k, v in mean.items()}
                    best, stage = escalate(model, task.scm, x, bg, mad, act, top, score)
                elif rule == "additive":
                    best, stage = escalate(model, task.scm, x, bg, mad, act, top,
                                           {p: F1[p[0]] + F1[p[1]] for p in topM_pairs})
                elif rule == "shap":
                    if explainer is None:
                        explainer = shap.explainers.Permutation(model, shap.maskers.Independent(bg[:100]))
                    phi = explainer(x[None, :], max_evals=max(500, 2 * len(x) + 1), silent=True).values[0]
                    order = sorted(act, key=lambda t: -abs(phi[t]))[:M]
                    best, stage = escalate(model, task.scm, x, bg, mad, act, order,
                                           {p: abs(phi[p[0]]) + abs(phi[p[1]])
                                            for p in combinations(order, 2)})
                elif rule == "random":
                    perm = list(qrng.permutation(len(topM_pairs)))
                    best, stage = escalate(model, task.scm, x, bg, mad, act, top,
                                           {topM_pairs[k]: -r for r, k in enumerate(perm)})
                elif rule == "topM_all":
                    best = search_sets(model, task.scm, x, bg, mad, topM_pairs); stage = "pair"
                    if not best:
                        best = search_sets(model, task.scm, x, bg, mad, combinations(top, 3))
                        stage = "triple" if best else ""
                else:  # all_pairs
                    best = search_sets(model, task.scm, x, bg, mad, combinations(act, 2)); stage = "pair"
                    if not best:
                        best = search_sets(model, task.scm, x, bg, mad, combinations(act, 3))
                        stage = "triple" if best else ""
                rows_out.append({
                    "dataset": dataset, "seed": seed, "model": mname, "query": q, "test_row": i,
                    "rule": rule, "success": best is not None, "stage": stage,
                    "cost": best[0] if best else np.nan,
                    "L_active": len(best[2]) if best else np.nan,
                    "actions": str(sorted(task.features[j] for j in best[2])) if best else "",
                    "evals": sum(model.counts.values()), "time": time.time() - t0})
        print(f"{dataset} seed{seed} {mname} done, hard so far {n_hard} "
              f"({time.time() - t_start:.0f}s)", flush=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding="utf-8-sig")


def scm_do(scm, x, V, M):
    return scm.do(x, V, M)


def summarize():
    import glob
    from scipy.stats import binomtest
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "E6_*_seed*.csv"))
                    if os.path.getsize(f) > 10], ignore_index=True)
    pd.set_option("display.width", 250)
    s = df.groupby(["dataset", "rule"]).agg(n=("success", "size"), success=("success", "mean"),
                                            evals=("evals", "mean")).join(
        df[df["success"]].groupby(["dataset", "rule"])[["cost", "L_active"]].mean())
    tot = df.groupby("rule").agg(n=("success", "size"), success=("success", "mean"),
                                 evals=("evals", "mean")).join(
        df[df["success"]].groupby("rule")[["cost", "L_active"]].mean())
    s.round(3).to_csv(os.path.join(OUT, "E6_summary.csv"), encoding="utf-8-sig")
    print(s.round(3).to_string()); print("\n[All datasets]\n", tot.round(3).reindex(RULES).to_string())
    w = df.pivot_table(index=["dataset", "seed", "model", "query"], columns="rule",
                       values="success", aggfunc="first").astype(bool)
    rows = []
    for r in RULES[1:]:
        a = int((w["br_cg4"] & ~w[r]).sum()); b = int((~w["br_cg4"] & w[r]).sum())
        rows.append({"vs": r, "br_only": a, "other_only": b,
                     "p": binomtest(a, a + b).pvalue if a + b else np.nan})
    mc = pd.DataFrame(rows)
    mc.to_csv(os.path.join(OUT, "E6_mcnemar.csv"), index=False, encoding="utf-8-sig")
    print("\n[br_cg4 vs other rules, exact McNemar]\n", mc.to_string(index=False))
    both = w["br_cg4"] & w["all_pairs"]
    c = df.pivot_table(index=["dataset", "seed", "model", "query"], columns="rule", values="cost",
                       aggfunc="first")
    print("\n[cost ratio br_cg4 / all_pairs on both-success]",
          round(float((c["br_cg4"][both] / c["all_pairs"][both]).median()), 3))


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    if ds == "summary":
        summarize(); sys.exit()
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for d in (DATASETS if ds == "all" else [ds]):
        run(d, seed)
