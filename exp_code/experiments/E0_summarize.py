"""
Summarise E0 results (exp_result/E0/E0_*.csv).

Tables (written to exp_result/E0/):
  E0_overall.csv        per method: success over all queries; cost, L0, L_active, evaluations,
                        time, cross-model validity, tail use over that method's successes
  E0_by_dataset.csv     same, per dataset
  E0_paired_vs_v2.csv   paired comparison with acbfi_v2 on queries where BOTH succeed:
                        means, Wilcoxon signed-rank p-values with Holm correction
  E0_v2_stage.csv       which stage of acbfi_v2 found the recourse
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "exp_result", "E0")
ORDER = ["acbfi_v2", "acbfi_v1", "untargeted", "lewis_style", "shap_staged", "shap_ce",
         "random_grid_ce"]
METRICS = ["cost", "L0", "L_active", "evals_total", "time"]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def describe(df, by):
    succ = df.groupby(by)["success"].mean().rename("success")
    s = df[df["success"]].groupby(by).agg(
        cost=("cost", "mean"), L0=("L0", "mean"), L_active=("L_active", "mean"),
        evals_diag=("evals_diag", "mean"), evals_search=("evals_search", "mean"),
        evals_total=("evals_total", "mean"), time=("time", "mean"),
        cross_valid=("cross_valid", "mean"), tail=("tail", "mean"))
    n = df.groupby(by).size().rename("n_queries")
    return pd.concat([n, succ, s], axis=1)


def main():
    files = [f for f in glob.glob(os.path.join(OUT, "E0_*_seed*.csv")) if os.path.getsize(f) > 10]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["key"] = (df["dataset"] + "|" + df["model"] + "|" + df["seed"].astype(str) + "|"
                 + df["query"].astype(str))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    overall = describe(df, "method").reindex(ORDER)
    by_ds = describe(df, ["dataset", "method"])
    overall.round(3).to_csv(os.path.join(OUT, "E0_overall.csv"), encoding="utf-8-sig")
    by_ds.round(3).to_csv(os.path.join(OUT, "E0_by_dataset.csv"), encoding="utf-8-sig")
    print("[Overall]\n", overall.round(3).to_string())

    ref = df[df["method"] == "acbfi_v2"].set_index("key")
    rows, pvals = [], []
    for m in ORDER[1:]:
        oth = df[df["method"] == m].set_index("key")
        both = ref.index[(ref["success"]) & oth.reindex(ref.index)["success"].fillna(False).astype(bool)]
        for met in METRICS:
            a, b = ref.loc[both, met].astype(float), oth.loc[both, met].astype(float)
            p = wilcoxon(a, b).pvalue if len(both) > 5 and not np.allclose(a, b) else np.nan
            rows.append({"method": m, "metric": met, "n_both": len(both),
                         "acbfi_v2": a.mean(), "other": b.mean(), "p_raw": p})
            pvals.append(p)
    paired = pd.DataFrame(rows)
    mask = paired["p_raw"].notna()
    paired.loc[mask, "p_holm"] = holm(paired.loc[mask, "p_raw"])
    paired.round(4).to_csv(os.path.join(OUT, "E0_paired_vs_v2.csv"), index=False, encoding="utf-8-sig")
    print("\n[Paired vs acbfi_v2, both-success]\n", paired.round(4).to_string(index=False))

    st = df[(df["method"] == "acbfi_v2") & df["success"]].groupby("dataset")["stage"] \
        .value_counts(normalize=True).unstack(fill_value=0).round(3)
    st.to_csv(os.path.join(OUT, "E0_v2_stage.csv"), encoding="utf-8-sig")
    print("\n[acbfi_v2 stage share]\n", st.to_string())
    print("\n[By dataset: success / cost / evals_total]\n",
          by_ds[["n_queries", "success", "cost", "L_active", "evals_total"]].round(3).to_string())


if __name__ == "__main__":
    main()
