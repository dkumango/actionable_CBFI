"""
E4. DAG sensitivity (review Major 6: LLM-elicited DAG is unvalidated).

For each dataset (seed s) the same models and queries as E0m are used. The SCM is rebuilt
under alternative structures (same features, same direction/bound constraints):
  llm        the LLM/domain DAG used in the paper (reference)
  drop30_rK  30 % of the edges removed at random (K = 1, 2)
  rev30_rK   30 % of the edges reversed at random, skipping reversals that create a cycle
  empty      no edges (independent features: no downstream propagation)
  pc         PC algorithm (Fisher-z, alpha = 0.05) on 2,000 training rows; undirected edges
             oriented along the canonical column order; edges INTO non-actionable
             features are dropped (an action cannot change age, history, ...)
  gbr        llm edges with nonlinear (gradient boosting) structural equations
Methods: acbfi_v3, untargeted. Per query and variant we record success, the direct
actions, and whether those direct actions still flip the model when propagated through
the REFERENCE (llm) SCM ('valid_under_llm') and with no propagation ('valid_no_prop').

Usage: python E4_dag_sensitivity.py <dataset|all> [seed]
Output: exp_result/E4/E4_<dataset>_seed<seed>.csv, E4_<dataset>_seed<seed>_edges.json
"""

import json
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../acbfi", "../real", "../synthetic"):
    sys.path.insert(0, os.path.join(HERE, p))
from core import THRESHOLD, CountingModel, acbfi_v3, untargeted  # noqa: E402
from R1_lever_vs_cause import make_models  # noqa: E402
from real_data import build_scm, load_task  # noqa: E402

E0M = os.path.join(HERE, "..", "..", "exp_result", "E0m")
OUT = os.path.join(HERE, "..", "..", "exp_result", "E4")
os.makedirs(OUT, exist_ok=True)
DATASETS = ["heloc", "cdc_diabetes", "german_credit", "adult", "diabetes", "financial_loan"]


def drop_edges(edges, frac, rng):
    keep = rng.permutation(len(edges))[: len(edges) - int(round(frac * len(edges)))]
    return [edges[i] for i in sorted(keep)]


def reverse_edges(edges, frac, rng):
    out = list(edges)
    for i in rng.permutation(len(edges))[: int(round(frac * len(edges)))]:
        cand = list(out)
        cand[i] = (out[i][1], out[i][0])
        if nx.is_directed_acyclic_graph(nx.DiGraph(cand)):
            out = cand
    return out


def pc_edges(task, rng, nonactionable):
    from causallearn.search.ConstraintBased.PC import pc
    X = task.X_train[rng.choice(len(task.X_train), min(2000, len(task.X_train)), replace=False)]
    X = X + rng.normal(0, 1e-6, X.shape)          # avoid singular correlation for constant cols
    g = pc(X, alpha=0.05, indep_test="fisherz", show_progress=False).G.graph
    F, d = task.features, len(task.features)
    edges = []
    for i in range(d):
        for j in range(d):
            if g[j, i] == 1 and g[i, j] == -1:                 # i -> j
                edges.append((i, j))
            elif i < j and g[i, j] == -1 and g[j, i] == -1:    # undirected: canonical order
                edges.append((i, j))
    edges = [(F[i], F[j]) for i, j in edges if F[j] not in nonactionable]
    G = nx.DiGraph(); G.add_nodes_from(F)
    for e in edges:                                        # keep acyclic
        G.add_edge(*e)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(*e)
    return list(G.edges())


def apply_actions(scm, x, x_cf, intervened):
    V = np.zeros((1, len(x))); M = np.zeros_like(V, bool)
    for j in intervened:
        V[0, j] = x_cf[j]; M[0, j] = True
    return scm.do(x, V, M)[0]


def run(dataset, seed=0):
    out_csv = os.path.join(OUT, f"E4_{dataset}_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"skip {out_csv}"); return
    t_start = time.time()
    e0 = pd.read_csv(os.path.join(E0M, f"E0_{dataset}_seed{seed}.csv"))
    queries = e0[["model", "query", "test_row"]].drop_duplicates()
    task = load_task(dataset, seed=seed)
    act = task.actionable
    nonact = {task.features[j] for j in range(len(task.features)) if j not in act}
    rng = np.random.default_rng(seed)
    bg = task.X_train[rng.choice(len(task.X_train), min(2000, len(task.X_train)), replace=False)]
    vrng = np.random.default_rng(1000 + seed)
    variants = {"llm": (task.edges, "linear")}
    for k in (1, 2):
        variants[f"drop30_r{k}"] = (drop_edges(task.edges, 0.3, vrng), "linear")
        variants[f"rev30_r{k}"] = (reverse_edges(task.edges, 0.3, vrng), "linear")
    variants["empty"] = ([], "linear")
    variants["pc"] = (pc_edges(task, vrng, nonact), "linear")
    variants["gbr"] = (task.edges, "gbr")
    with open(out_csv.replace(".csv", "_edges.json"), "w", encoding="utf-8") as f:
        json.dump({k: {"edges": [list(e) for e in v[0]], "eq": v[1]} for k, v in variants.items()},
                  f, indent=1)
    scms = {k: build_scm(task, e, r) for k, (e, r) in variants.items()}
    ref, nop = scms["llm"], scms["empty"]
    fns = {}
    for name, clf in make_models().items():
        clf.fit(task.X_train, task.y_train)
        fns[name] = (lambda X, c=clf: c.predict_proba(np.asarray(X))[:, 1])
    rows = []
    for mname, sub in queries.groupby("model", sort=False):
        model = CountingModel(fns[mname])
        for _, qr in sub.iterrows():
            q, i = int(qr["query"]), int(qr["test_row"])
            x = task.X_test[i]
            for vname, scm in scms.items():
                for meth, fn in (("acbfi_v3", acbfi_v3), ("untargeted", untargeted)):
                    r = fn(model, scm, x, bg, act, rng=np.random.default_rng(seed * 100000 + q))
                    rec = {"dataset": dataset, "seed": seed, "model": mname, "query": q,
                           "test_row": i, "variant": vname, "method": meth, "success": r.success,
                           "evals": r.evals_diag + r.evals_search, "actions": "",
                           "valid_under_llm": np.nan, "valid_no_prop": np.nan}
                    if r.success:
                        rec["actions"] = json.dumps(sorted(task.features[j] for j in r.intervened))
                        xr = apply_actions(ref, x, r.x_cf, r.intervened)
                        xn = apply_actions(nop, x, r.x_cf, r.intervened)
                        rec["valid_under_llm"] = bool(fns[mname](xr[None])[0] < THRESHOLD)
                        rec["valid_no_prop"] = bool(fns[mname](xn[None])[0] < THRESHOLD)
                    rows.append(rec)
        print(f"{dataset} seed{seed} {mname} done ({time.time() - t_start:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def summarize():
    import glob
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "E4_*_seed*.csv"))],
                   ignore_index=True)
    df["key"] = df["dataset"] + "|" + df["model"] + "|" + df["seed"].astype(str) + "|" + \
        df["query"].astype(str) + "|" + df["method"]
    ref = df[df["variant"] == "llm"].set_index("key")["actions"]
    df["same_actions_as_llm"] = np.where(df["success"],
                                         df["actions"] == df["key"].map(ref), np.nan)
    pd.set_option("display.width", 250)
    s = df.groupby(["dataset", "variant", "method"]).agg(
        success=("success", "mean"), same_actions=("same_actions_as_llm", "mean"),
        valid_under_llm=("valid_under_llm", "mean"), valid_no_prop=("valid_no_prop", "mean"),
        evals=("evals", "mean")).round(3)
    s.to_csv(os.path.join(OUT, "E4_summary.csv"), encoding="utf-8-sig")
    print(s.to_string())
    # does the v3 vs untargeted comparison survive each DAG variant?
    w = df.pivot_table(index=["dataset", "variant", "model", "seed", "query"], columns="method",
                       values="success", aggfunc="first").astype(bool).reset_index()
    g = w.groupby(["dataset", "variant"]).apply(
        lambda t: pd.Series({"v3_only": int((t.acbfi_v3 & ~t.untargeted).sum()),
                             "untargeted_only": int((~t.acbfi_v3 & t.untargeted).sum()),
                             "succ_v3": t.acbfi_v3.mean(), "succ_untargeted": t.untargeted.mean()}))
    g.round(3).to_csv(os.path.join(OUT, "E4_v3_vs_untargeted.csv"), encoding="utf-8-sig")
    print("\n[v3 vs untargeted under each DAG variant]\n", g.round(3).to_string())


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    if ds == "summary":
        summarize(); sys.exit()
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for d in (DATASETS if ds == "all" else [ds]):
        run(d, seed)
