"""
E8. Using both CBFI groups prescriptively: C_G3 (common / redundant support of the adverse
label -> joint necessity) and C_G4 (cooperative support -> substitutable single levers).

For a query x and background rows r (values clipped to the allowed direction/bounds by the
SCM), with s_S(r) = 1[f(x^{do(S = r_S)}) < tau]:
  PG1(i)    = P(s_i = 1)                               single lever i alone suffices
  PG3(i,j)  = P(s_ij = 1, s_i = 0, s_j = 0)            joint necessity (redundant support)
  PG4(i,j)  = P(s_i = 1, s_j = 1)                      substitutable levers (cooperative support)

E8-a  synthetic OR / AND structures: do PG3 / PG4 recover redundant vs cooperative support?
E8-b  escalation stage (queries no single grid action solves): pair rule PG3 vs C_G4^br vs
      additive vs top-M exhaustive (all 10 datasets, 3 seeds)
E8-c  substitutable levers: on single-solvable queries, how many distinct single levers
      succeed on the grid, the cost premium of the second-best one, and whether PG4 with the
      best lever predicts which alternatives succeed
E8-d  difficulty prediction: do max PG1 and max PG3 (30 rows) predict whether a query is
      single-solvable / needs escalation / unsolved (AUC)?

Usage: python E8_cg3_cg4.py synth | <dataset|all> [seed] | summary
Output: exp_result/E8/
"""

import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../acbfi", "../real", "../synthetic"):
    sys.path.insert(0, os.path.join(HERE, p))
from core import THRESHOLD, CountingModel, _search_targets, grid_v1, mad_of  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "exp_result", "E8")
E0M = os.path.join(HERE, "..", "..", "exp_result", "E0m")
os.makedirs(OUT, exist_ok=True)
DATASETS = ["heloc", "adult", "german_credit", "diabetes", "taiwan_default", "gmsc",
            "student_dropout", "student_perf", "financial_loan", "cdc_diabetes"]
BINS, M, P, T, NROWS = 6, 6, 3, 2, 30


# ------------------------------------------------------------------ shared --------------
def single_grid(model, scm, x, bg, act, mad):
    """Step 1 on the grid: per feature best gain, success flag and cheapest cost."""
    p0 = model(x[None])[0]
    F1, cost = {}, {}
    for i in act:
        g = grid_v1(bg, i, BINS, x, scm)
        if len(g) == 0:
            F1[i], cost[i] = 0.0, np.inf
            continue
        V = np.zeros((len(g), len(x))); Mk = np.zeros_like(V, bool)
        V[:, i] = g; Mk[:, i] = True
        Xcf = scm.do(x, V, Mk)
        p = model(Xcf)
        F1[i] = max(0.0, float((p0 - p).max()))
        ok = p < THRESHOLD
        c = (np.abs(Xcf - x) / mad).sum(axis=1)
        cost[i] = float(c[ok].min()) if ok.any() else np.inf
    return F1, cost


def flip_matrix(model, scm, x, rows, feats):
    """s_i(r) for each feature, s_ij(r) for each pair (same rows)."""
    s1 = {}
    for i in feats:
        Mk = np.zeros(rows.shape, bool); Mk[:, i] = True
        s1[i] = model(scm.do(x, rows, Mk)) < THRESHOLD
    s2, gain2 = {}, {}
    p0 = model(x[None])[0]
    for i, j in combinations(feats, 2):
        Mk = np.zeros(rows.shape, bool); Mk[:, [i, j]] = True
        p = model(scm.do(x, rows, Mk))
        s2[(i, j)] = p < THRESHOLD
        gain2[(i, j)] = float(max(0.0, (p0 - p).max()))
    return s1, s2, gain2


def pg(s1, s2):
    pg1 = {i: float(v.mean()) for i, v in s1.items()}
    pg3 = {k: float((v & ~s1[k[0]] & ~s1[k[1]]).mean()) for k, v in s2.items()}
    pg4 = {k: float((s1[k[0]] & s1[k[1]]).mean()) for k in s2}
    return pg1, pg3, pg4


def escalate(model, scm, x, bg, mad, top, score):
    pairs = sorted(combinations(top, 2), key=lambda p: -score.get(p, score.get((p[1], p[0]), 0)))
    best = None
    for s in pairs[:P]:
        r = _search_targets(model, scm, x, list(s), bg, mad, BINS)
        if r and (best is None or r[0] < best[0]):
            best = r
    if best:
        return best, "pair"
    for p in pairs[:T]:
        rest = [t for t in top if t not in p]
        if not rest:
            continue
        sc = lambda a, b: score.get((a, b), score.get((b, a), 0))
        third = max(rest, key=lambda t: max(sc(p[0], t), sc(p[1], t)))
        r = _search_targets(model, scm, x, list(p) + [third], bg, mad, BINS)
        if r and (best is None or r[0] < best[0]):
            best = r
    return (best, "triple") if best else (None, "")


def topm_all(model, scm, x, bg, mad, top):
    best = None
    for size, stage in ((2, "pair"), (3, "triple")):
        for s in combinations(top, size):
            r = _search_targets(model, scm, x, list(s), bg, mad, BINS)
            if r and (best is None or r[0] < best[0]):
                best = r
        if best:
            return best, stage
    return None, ""


# ------------------------------------------------------------------ E8-a ----------------
def synth():
    from synth_lib import SCM, sigmoid
    rng = np.random.default_rng(11)
    rows_out = []
    for structure in ("OR", "AND"):
        for beta in (1.0, 2.0, 4.0):
            scm = SCM(6, {}, {})
            pop = scm.sample(20000, rng)
            if structure == "OR":     # adverse if a OR b is high: redundant support
                z = lambda X, b=beta: b * np.maximum(X[:, 0], X[:, 1]) + 0.3 * X[:, 2:6].sum(axis=1)
            else:                     # adverse only if a AND b are high: cooperative support
                z = lambda X, b=beta: b * np.minimum(X[:, 0], X[:, 1]) + 0.3 * X[:, 2:6].sum(axis=1)
            c = -np.quantile(z(pop), 0.7)
            model = CountingModel(lambda X, z=z, c=c: sigmoid(c + z(X)))
            bg = pop[:3000]
            cand = np.where((model(pop) > 0.6) & (pop[:, 0] > 0.8) & (pop[:, 1] > 0.8))[0]
            for q, i in enumerate(rng.choice(cand, size=min(80, len(cand)), replace=False)):
                x = pop[i]
                rows = bg[rng.choice(len(bg), 60, replace=False)]
                s1, s2, gain2 = flip_matrix(model, scm, x, rows, [0, 1])
                pg1, pg3, pg4 = pg(s1, s2)
                mad = mad_of(bg)
                F1, cost = single_grid(model, scm, x, bg, [0, 1], mad)
                both = _search_targets(model, scm, x, [0, 1], bg, mad, BINS)
                rows_out.append({"structure": structure, "beta": beta, "query": q,
                                 "pg1_a": pg1[0], "pg1_b": pg1[1], "pg3": pg3[(0, 1)],
                                 "pg4": pg4[(0, 1)], "br_cg4": gain2[(0, 1)] - F1[0] - F1[1],
                                 "single_ok": bool(np.isfinite(cost[0]) or np.isfinite(cost[1])),
                                 "pair_ok": both is not None,
                                 "n_single_ok": int(np.isfinite(cost[0])) + int(np.isfinite(cost[1]))})
    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(OUT, "E8a_synthetic.csv"), index=False, encoding="utf-8-sig")
    df["pred"] = np.where(df.pg3 > df.pg4, "OR", "AND")
    df["pred_br"] = np.where(df.br_cg4 > 0, "OR", "AND")
    s = df.groupby(["structure", "beta"]).agg(pg3=("pg3", "mean"), pg4=("pg4", "mean"),
                                              br_cg4=("br_cg4", "mean"),
                                              acc_pg=("pred", lambda v: np.nan),
                                              single_ok=("single_ok", "mean"),
                                              n_single_ok=("n_single_ok", "mean"),
                                              pair_ok=("pair_ok", "mean"))
    s["acc_pg"] = df.assign(c=df.pred == df.structure).groupby(["structure", "beta"]).c.mean()
    s["acc_br"] = df.assign(c=df.pred_br == df.structure).groupby(["structure", "beta"]).c.mean()
    s.round(3).to_csv(os.path.join(OUT, "E8a_summary.csv"), encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(s.round(3).to_string())


# ------------------------------------------------------------------ E8-b/c/d -------------
def run(dataset, seed):
    from R1_lever_vs_cause import make_models
    from real_data import load_task
    out_csv = os.path.join(OUT, f"E8_{dataset}_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"skip {out_csv}"); return
    t0 = time.time()
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
    recs = []
    for mname, sub in queries.groupby("model", sort=False):
        model = CountingModel(fns[mname])
        for _, qr in sub.iterrows():
            q, i = int(qr["query"]), int(qr["test_row"])
            x = task.X_test[i]
            qrng = np.random.default_rng(seed * 100000 + q)
            F1, cost = single_grid(model, task.scm, x, bg, act, mad)
            top = sorted(act, key=lambda t: -F1[t])[:M]
            rows = bg[qrng.choice(len(bg), NROWS, replace=False)]
            model.reset()
            s1, s2, gain2 = flip_matrix(model, task.scm, x, rows, top)
            pg_evals = sum(model.counts.values())
            pg1, pg3, pg4 = pg(s1, s2)
            base = {"dataset": dataset, "seed": seed, "model": mname, "query": q,
                    "max_pg1": max(pg1.values()) if pg1 else 0.0,
                    "max_pg3": max(pg3.values()) if pg3 else 0.0,
                    "max_pg4": max(pg4.values()) if pg4 else 0.0,
                    "mean_pg3": float(np.mean(list(pg3.values()))) if pg3 else 0.0,
                    "max_F1": max(F1.values()) if F1 else 0.0, "pg_evals": pg_evals}
            ok_single = [t for t in act if np.isfinite(cost[t])]
            if ok_single:                                            # E8-c
                cs = sorted(cost[t] for t in ok_single)
                best = min(ok_single, key=lambda t: cost[t])
                alt = [t for t in top if t != best]
                pred = {t: pg4.get(tuple(sorted((best, t), key=top.index)), 0.0) if best in top else np.nan
                        for t in alt}
                recs.append({**base, "kind": "single", "n_alternatives": len(ok_single),
                             "cost_best": cs[0], "cost_second": cs[1] if len(cs) > 1 else np.nan,
                             "pg4_pred": str({task.features[t]: round(v, 3) for t, v in pred.items()}),
                             "alt_truth": str({task.features[t]: bool(np.isfinite(cost[t])) for t in alt})})
                continue
            # E8-b: hard query
            res = {}
            for rule in ("pg3", "br_cg4", "additive", "topM_all"):
                model.reset()
                if rule == "pg3":
                    best_, st_ = escalate(model, task.scm, x, bg, mad, top, pg3)
                    extra = pg_evals
                elif rule == "br_cg4":
                    sc = {k: gain2[k] for k in gain2}
                    best_, st_ = escalate(model, task.scm, x, bg, mad, top, sc)
                    extra = pg_evals
                elif rule == "additive":
                    best_, st_ = escalate(model, task.scm, x, bg, mad, top,
                                          {p: F1[p[0]] + F1[p[1]] for p in combinations(top, 2)})
                    extra = 0
                else:
                    best_, st_ = topm_all(model, task.scm, x, bg, mad, top)
                    extra = 0
                res[rule] = (best_, st_, sum(model.counts.values()) + extra)
            for rule, (best_, st_, ev) in res.items():
                recs.append({**base, "kind": "hard", "rule": rule, "success": best_ is not None,
                             "stage": st_, "evals": ev,
                             "cost": best_[0] if best_ else np.nan,
                             "topM_success": res["topM_all"][0] is not None})
        print(f"{dataset} seed{seed} {mname} done ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(recs).to_csv(out_csv, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ summary -------------
def summarize():
    import glob
    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "E8_*_seed*.csv"))
                    if os.path.getsize(f) > 10], ignore_index=True)
    pd.set_option("display.width", 220)
    hard = df[df.kind == "hard"]
    s = hard.groupby(["dataset", "rule"]).agg(n=("success", "size"), success=("success", "mean"),
                                             evals=("evals", "mean")).round(3)
    tot = hard.groupby("rule").agg(n=("success", "size"), success=("success", "mean"),
                                   evals=("evals", "mean")).round(3)
    w = hard.pivot_table(index=["dataset", "seed", "model", "query"], columns="rule",
                         values="success", aggfunc="first").astype(bool)
    mc = []
    for r in ("br_cg4", "additive", "topM_all"):
        a = int((w.pg3 & ~w[r]).sum()); b = int((~w.pg3 & w[r]).sum())
        mc.append({"pg3_vs": r, "pg3_only": a, "other_only": b,
                   "p": binomtest(a, a + b).pvalue if a + b else np.nan})
    single = df[df.kind == "single"]
    c = {"n_single_queries": len(single),
         "share_with_>=2_alternatives": float((single.n_alternatives >= 2).mean()),
         "median_alternatives": float(single.n_alternatives.median()),
         "median_cost_premium_2nd": float((single.cost_second / single.cost_best).median())}
    by_c = single.groupby("dataset").agg(n=("n_alternatives", "size"),
                                         ge2=("n_alternatives", lambda v: (v >= 2).mean()),
                                         median_alt=("n_alternatives", "median")).round(3)
    # E8-c: does PG4 with the best lever predict which alternatives succeed?
    ys, ps = [], []
    for _, r in single.iterrows():
        pred, truth = eval(r.pg4_pred, {"nan": np.nan}), eval(r.alt_truth)
        for k, v in pred.items():
            if k in truth and not pd.isna(v):
                ys.append(int(truth[k])); ps.append(v)
    auc_c = roc_auc_score(ys, ps) if len(set(ys)) > 1 else np.nan
    # E8-d: difficulty prediction
    q = df.drop_duplicates(["dataset", "seed", "model", "query"]).copy()
    hard_first = hard.drop_duplicates(["dataset", "seed", "model", "query"]).set_index(
        ["dataset", "seed", "model", "query"])["topM_success"]
    q["label"] = np.where(q.kind == "single", "single",
                          q.set_index(["dataset", "seed", "model", "query"]).index.map(
                              lambda k: "escalation" if hard_first.get(k, False) else "unsolved"))
    d = {"auc_single_from_maxPG1": roc_auc_score(q.label == "single", q.max_pg1),
         "auc_single_from_maxF1": roc_auc_score(q.label == "single", q.max_F1)}
    hq = q[q.label != "single"]
    if hq.label.nunique() > 1:
        d["auc_escalation_vs_unsolved_from_maxPG3"] = roc_auc_score(hq.label == "escalation", hq.max_pg3)
        d["auc_escalation_vs_unsolved_from_meanPG3"] = roc_auc_score(hq.label == "escalation", hq.mean_pg3)
    d["label_counts"] = q.label.value_counts().to_dict()
    s.to_csv(os.path.join(OUT, "E8b_by_dataset.csv"), encoding="utf-8-sig")
    pd.DataFrame(mc).to_csv(os.path.join(OUT, "E8b_mcnemar.csv"), index=False, encoding="utf-8-sig")
    by_c.to_csv(os.path.join(OUT, "E8c_by_dataset.csv"), encoding="utf-8-sig")
    import json
    json.dump({"E8c": c, "E8c_auc_pg4": auc_c, "E8d": d}, open(os.path.join(OUT, "E8cd_summary.json"), "w"),
              indent=1, default=float)
    print("[E8-b by dataset]\n", s.to_string()); print("\n[E8-b total]\n", tot.to_string())
    print("\n[E8-b McNemar]\n", pd.DataFrame(mc).to_string(index=False))
    print("\n[E8-c]", c, "\nAUC(PG4 with best lever -> alternative succeeds):", auc_c, "\n", by_c.to_string())
    print("\n[E8-d]", d)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "synth"
    if arg == "synth":
        synth()
    elif arg == "summary":
        summarize()
    else:
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        for dname in (DATASETS if arg == "all" else [arg]):
            run(dname, seed)
