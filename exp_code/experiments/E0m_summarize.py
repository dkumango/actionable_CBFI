"""
Summarise the multi-seed constrained benchmark (exp_result/E0m/E0_<dataset>_seed<s>.csv).

Writes to exp_result/E0m/:
  E0m_by_dataset.csv   per dataset x method: success, cost, cost_action, L_active,
                       evals_total, tail, cross_valid as mean over seeds (+ sd of per-seed means)
  E0m_mcnemar.csv      per dataset: queries where acbfi_v3 succeeds and the other fails vs the
                       reverse, exact McNemar p (pooled over seeds and models)
  E0m_paired_cost.csv  per dataset: cost / cost_action / L_active / evals on queries where
                       acbfi_v3 and the other method both succeed, Wilcoxon p (Holm)
  E0m_v3_stage.csv     share of acbfi_v3 successes by stage
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from E0_summarize import holm  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "exp_result", "E0m")
ORDER = ["acbfi_v3", "acbfi_v2", "acbfi_v1", "untargeted", "lewis_style", "shap_staged",
         "shap_ce", "random_grid_ce"]
MET = ["cost", "cost_action", "L_active", "evals_total", "tail", "cross_valid"]


def main():
    files = [f for f in glob.glob(os.path.join(OUT, "E0_*_seed*.csv")) if os.path.getsize(f) > 10]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["key"] = (df["dataset"] + "|" + df["model"] + "|" + df["seed"].astype(str) + "|"
                 + df["query"].astype(str))
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print("seeds per dataset:", df.groupby("dataset")["seed"].unique().apply(sorted).to_dict())

    per_seed = df.groupby(["dataset", "method", "seed"]).agg(success=("success", "mean")).join(
        df[df["success"]].groupby(["dataset", "method", "seed"])[MET].mean())
    mean = per_seed.groupby(["dataset", "method"]).mean()
    sd = per_seed.groupby(["dataset", "method"])[["success", "cost", "evals_total"]].std() \
        .add_suffix("_sd")
    n = df.groupby(["dataset", "method"]).size().rename("n_queries")
    by_ds = pd.concat([n, mean, sd], axis=1)
    by_ds.round(3).to_csv(os.path.join(OUT, "E0m_by_dataset.csv"), encoding="utf-8-sig")
    show = by_ds[["n_queries", "success", "success_sd", "cost", "cost_action", "L_active",
                  "evals_total"]].round(3)
    print("\n[By dataset x method (mean over seeds)]\n", show.to_string())

    rows_m, rows_c = [], []
    for ds, sub in df.groupby("dataset"):
        w = sub.pivot_table(index="key", columns="method", values="success", aggfunc="first").astype(bool)
        for m in ORDER[1:]:
            if m not in w:
                continue
            a = int((w["acbfi_v3"] & ~w[m]).sum()); b = int((~w["acbfi_v3"] & w[m]).sum())
            p = binomtest(a, a + b).pvalue if a + b > 0 else np.nan
            rows_m.append({"dataset": ds, "method": m, "v3_only": a, "other_only": b, "p": p})
            ref = sub[sub["method"] == "acbfi_v3"].set_index("key")
            oth = sub[sub["method"] == m].set_index("key")
            both = ref.index[ref["success"] & oth.reindex(ref.index)["success"].fillna(False).astype(bool)]
            for met in ["cost", "cost_action", "L_active", "evals_total"]:
                x, y = ref.loc[both, met].astype(float), oth.loc[both, met].astype(float)
                p2 = wilcoxon(x, y).pvalue if len(both) > 5 and not np.allclose(x, y) else np.nan
                rows_c.append({"dataset": ds, "method": m, "metric": met, "n_both": len(both),
                               "acbfi_v3": x.mean(), "other": y.mean(), "p_raw": p2})
    mc = pd.DataFrame(rows_m)
    pc = pd.DataFrame(rows_c)
    ok = pc["p_raw"].notna()
    pc.loc[ok, "p_holm"] = holm(pc.loc[ok, "p_raw"])
    mc.to_csv(os.path.join(OUT, "E0m_mcnemar.csv"), index=False, encoding="utf-8-sig")
    pc.round(4).to_csv(os.path.join(OUT, "E0m_paired_cost.csv"), index=False, encoding="utf-8-sig")
    print("\n[McNemar: acbfi_v3 vs others, pooled over seeds and models]\n",
          mc.pivot_table(index="dataset", columns="method",
                         values=["v3_only", "other_only"]).to_string())
    print("\n p-values:\n", mc.pivot_table(index="dataset", columns="method", values="p")
          .map(lambda v: f"{v:.1e}").to_string())
    print("\n[Both-success cost vs untargeted / lewis]\n",
          pc[pc["method"].isin(["untargeted", "lewis_style", "acbfi_v1"])
             & (pc["metric"].isin(["cost", "cost_action", "L_active"]))].round(3).to_string(index=False))

    st = df[(df["method"] == "acbfi_v3") & df["success"]].groupby("dataset")["stage"] \
        .value_counts(normalize=True).unstack(fill_value=0).round(3)
    st.to_csv(os.path.join(OUT, "E0m_v3_stage.csv"), encoding="utf-8-sig")
    print("\n[acbfi_v3 stage share]\n", st.to_string())


if __name__ == "__main__":
    main()
