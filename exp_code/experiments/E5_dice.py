"""
E5. DiCE baseline (Mothilal et al., FAT* 2020; dice-ml 0.12, model-agnostic 'random' method)
on exactly the E0m queries (same seeds, splits, models, background).

DiCE is non-causal: it changes features independently (no SCM propagation). Actionability
is enforced with features_to_vary = directly actionable features, and the direction /
bound constraints of real_data.DIRECTIONS / BOUNDS through a per-query permitted_range.
One counterfactual per query (total_CFs=1), desired class = favourable (0).
Metrics as in E0: success (re-checked with our model), MAD-L1 cost, L0 (= L_active),
evaluations (every predict_proba row DiCE requests), time, tail use, cross-model validity.

Usage: python E5_dice.py <dataset|all> [seed]
Output: exp_result/E5/E5_dice_<dataset>_seed<seed>.csv
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("../acbfi", "../real", "../synthetic"):
    sys.path.insert(0, os.path.join(HERE, p))
from core import THRESHOLD, mad_of  # noqa: E402
from R1_lever_vs_cause import make_models  # noqa: E402
from real_data import load_task  # noqa: E402

E0M = os.path.join(HERE, "..", "..", "exp_result", "E0m")
OUT = os.path.join(HERE, "..", "..", "exp_result", "E5")
os.makedirs(OUT, exist_ok=True)
DATASETS = ["heloc", "cdc_diabetes", "german_credit", "adult", "diabetes", "financial_loan"]


class CountingSklearn:
    """sklearn-style wrapper around p_adv(X) that counts every evaluated row."""

    def __init__(self, fn, features):
        self.fn, self.features, self.n = fn, features, 0
        self.classes_ = np.array([0, 1])

    def _arr(self, X):
        if isinstance(X, pd.DataFrame):
            X = X[self.features]
        return np.asarray(X, float)

    def predict_proba(self, X):
        A = self._arr(X)
        self.n += len(A)
        p = self.fn(A)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= THRESHOLD).astype(int)


def run(dataset, seed=0, timeout_s=60):
    import dice_ml
    out_csv = os.path.join(OUT, f"E5_dice_{dataset}_seed{seed}.csv")
    if os.path.exists(out_csv):
        print(f"skip {out_csv}"); return
    t_start = time.time()
    e0 = pd.read_csv(os.path.join(E0M, f"E0_{dataset}_seed{seed}.csv"))
    queries = e0[["model", "query", "test_row"]].drop_duplicates()
    task = load_task(dataset, seed=seed)
    F, act = task.features, task.actionable
    rng = np.random.default_rng(seed)
    bg = task.X_train[rng.choice(len(task.X_train), min(2000, len(task.X_train)), replace=False)]
    mad = mad_of(bg)
    lo, hi = np.quantile(task.X_train, 0.025, axis=0), np.quantile(task.X_train, 0.975, axis=0)
    fmin, fmax = task.X_train.min(axis=0), task.X_train.max(axis=0)
    train_df = pd.DataFrame(task.X_train, columns=F)
    train_df["y"] = task.y_train
    data = dice_ml.Data(dataframe=train_df, continuous_features=list(F), outcome_name="y")
    fns = {}
    for name, clf in make_models().items():
        clf.fit(task.X_train, task.y_train)
        if hasattr(clf, "n_jobs"):      # DiCE predicts many tiny batches: avoid joblib overhead
            clf.set_params(n_jobs=1)
        fns[name] = (lambda X, c=clf: c.predict_proba(np.asarray(X))[:, 1])
    dirs = getattr(task.scm, "directions", {})
    bnds = getattr(task.scm, "bounds", {})
    rows = []
    for mname, sub in queries.groupby("model", sort=False):
        wrapper = CountingSklearn(fns[mname], list(F))
        dmodel = dice_ml.Model(model=wrapper, backend="sklearn", model_type="classifier")
        explainer = dice_ml.Dice(data, dmodel, method="random")
        others = [f for n, f in fns.items() if n != mname]
        for _, qr in sub.iterrows():
            q, i = int(qr["query"]), int(qr["test_row"])
            x = task.X_test[i]
            prange = {}
            for j in act:
                a, b = float(fmin[j]), float(fmax[j])
                if dirs.get(j, 0) > 0:
                    a = float(x[j])
                elif dirs.get(j, 0) < 0:
                    b = float(x[j])
                if j in bnds:
                    a, b = max(a, min(bnds[j][0], x[j])), min(b, max(bnds[j][1], x[j]))
                prange[F[j]] = [a, max(a, b)]
            wrapper.n = 0
            t0 = time.time()
            rec = {"dataset": dataset, "seed": seed, "model": mname, "query": q, "test_row": i,
                   "method": "dice_random", "success": False, "cost": np.nan, "L0": np.nan,
                   "L_active": np.nan, "evals_total": 0, "time": np.nan, "tail": np.nan,
                   "cross_valid": np.nan, "changed": ""}
            try:
                res = explainer.generate_counterfactuals(
                    pd.DataFrame([x], columns=F), total_CFs=1, desired_class=0,
                    features_to_vary=[F[j] for j in act], permitted_range=prange,
                    random_seed=seed * 100000 + q, verbose=False)
                cfs = res.cf_examples_list[0].final_cfs_df
                if cfs is not None and len(cfs):
                    xcf = cfs[F].to_numpy(float)[0]
                    if fns[mname](xcf[None])[0] < THRESHOLD:
                        ch = np.abs(xcf - x) > 1e-6
                        rec.update(success=True, cost=float((np.abs(xcf - x) / mad).sum()),
                                   L0=int(ch.sum()), L_active=int(ch.sum()),
                                   tail=bool(np.any(ch & ((xcf < lo) | (xcf > hi)))),
                                   cross_valid=float(np.mean([f(xcf[None])[0] < THRESHOLD
                                                              for f in others])),
                                   changed=str([F[j] for j in np.where(ch)[0]]))
            except Exception as e:                       # DiCE raises when nothing is found
                rec["changed"] = f"error: {type(e).__name__}"
            rec["evals_total"] = wrapper.n
            rec["time"] = time.time() - t0
            rows.append(rec)
        print(f"{dataset} seed{seed} {mname} done ({time.time() - t_start:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


def summarize():
    import glob
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "E5_dice_*_seed*.csv"))],
                  ignore_index=True)
    e = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(E0M, "E0_*_seed*.csv"))],
                  ignore_index=True)
    e = e[e["method"].isin(["acbfi_v3", "untargeted", "shap_ce", "random_grid_ce"])]
    allm = pd.concat([e, d], ignore_index=True)
    allm = allm[allm["dataset"].isin(d["dataset"].unique())]
    s = allm.groupby(["dataset", "method"]).agg(success=("success", "mean")).join(
        allm[allm["success"]].groupby(["dataset", "method"])[
            ["cost", "L_active", "evals_total", "tail", "cross_valid"]].mean())
    s.round(3).to_csv(os.path.join(OUT, "E5_summary.csv"), encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
    print("\nDiCE errors:", d["changed"].str.startswith("error").sum(), "of", len(d))


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    if ds == "summary":
        summarize(); sys.exit()
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for dname in (DATASETS if ds == "all" else [ds]):
        run(dname, seed)
