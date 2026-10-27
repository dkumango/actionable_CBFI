"""
Build every table, figure and in-text number of the revised manuscript from the result
files, so the manuscript can be regenerated whenever results change (e.g. extra E7 seeds).

Output: exp_result/manuscript_assets/assets.json and fig*.png
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
RES = os.path.join(ROOT, "exp_result")
OUT = os.path.join(ROOT, "exp_result", "manuscript_assets")
os.makedirs(OUT, exist_ok=True)
for p in ("../acbfi", "../real", "../synthetic"):
    sys.path.insert(0, os.path.join(HERE, p))

DS = ["heloc", "german_credit", "taiwan_default", "gmsc", "adult", "financial_loan",
      "cdc_diabetes", "diabetes", "student_dropout", "student_perf"]
NAMES = {"heloc": "HELOC", "german_credit": "German Credit", "taiwan_default": "Taiwan Default",
         "gmsc": "Give Me Some Credit", "adult": "Adult Income", "financial_loan": "Lending Loan",
         "cdc_diabetes": "CDC Diabetes", "diabetes": "Pima Diabetes",
         "student_dropout": "Student Dropout", "student_perf": "Student Performance"}
DOMAIN = {"heloc": "Credit", "german_credit": "Credit", "taiwan_default": "Credit",
          "gmsc": "Credit", "adult": "Income", "financial_loan": "Credit",
          "cdc_diabetes": "Health", "diabetes": "Health", "student_dropout": "Education",
          "student_perf": "Education"}
ADVERSE = {"heloc": "RiskPerformance = Bad", "german_credit": "default = bad",
           "taiwan_default": "default next month", "gmsc": "distress within 2 years",
           "adult": "income <= 50K", "financial_loan": "Charged Off",
           "cdc_diabetes": "diabetes", "diabetes": "Outcome = 1",
           "student_dropout": "Dropout", "student_perf": "final grade G3 < 10"}
SOURCE = {"heloc": "FICO / OpenML 45553", "german_credit": "UCI", "taiwan_default": "UCI 350",
          "gmsc": "OpenML 46929", "adult": "UCI", "financial_loan": "Kaggle",
          "cdc_diabetes": "UCI 891", "diabetes": "UCI", "student_dropout": "UCI 697",
          "student_perf": "UCI 320"}
A = {}


def f3(v):
    """Unrounded float (rounding happens once, when the manuscript prints the value)."""
    return None if pd.isna(v) else float(v)


CAUSAL = ["untargeted", "acbfi_v1", "lewis_style", "shap_staged"]
NONCAUSAL = ["shap_ce", "dice"]


# ---------------------------------------------------------------- datasets ---------------
def datasets_table():
    from real_data import load_task
    rows = []
    for d in DS:
        t = load_task(d, 0)
        rows.append({"dataset": NAMES[d], "domain": DOMAIN[d], "n": int(t.n_raw),
                     "d": len(t.features), "actionable": len(t.actionable),
                     "edges": len(t.edges), "adverse": ADVERSE[d], "source": SOURCE[d],
                     "constrained": bool(getattr(t.scm, "directions", {}))})
    A["datasets"] = rows


# ---------------------------------------------------------------- main benchmark ---------
def main_benchmark():
    e = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E0m", "E0_*_seed*.csv"))],
                  ignore_index=True)
    dice = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E5", "E5_dice_*.csv"))],
                     ignore_index=True)
    meth = ["acbfi_v3", "acbfi_v1", "untargeted", "lewis_style", "shap_staged", "shap_ce"]
    rows = []
    for d in DS:
        sub = e[e.dataset == d]
        r = {"dataset": NAMES[d], "n_queries": int((sub.method == "acbfi_v3").sum())}
        for m in meth:
            s = sub[sub.method == m]
            r[f"succ_{m}"] = f3(s.success.mean())
            ok = s[s.success]
            r[f"Lact_{m}"] = f3(ok.L_active.mean())
            r[f"evals_{m}"] = f3(s.evals_total.mean())
            r[f"cost_{m}"] = f3(ok.cost.mean())
            r[f"costact_{m}"] = f3(ok.cost_action.mean())
        ds_dice = dice[dice.dataset == d]
        r["succ_dice"] = f3(ds_dice.success.mean()) if len(ds_dice) else None
        r["Lact_dice"] = f3(ds_dice[ds_dice.success].L_active.mean()) if len(ds_dice) else None
        r["evals_dice"] = f3(ds_dice.evals_total.mean()) if len(ds_dice) else None
        # McNemar v3 vs every baseline (DiCE joined on the same queries)
        w = sub.pivot_table(index=["model", "seed", "query"], columns="method", values="success",
                            aggfunc="first").astype(bool)
        if len(ds_dice):
            w["dice"] = ds_dice.set_index(["model", "seed", "query"]).success.astype(bool).reindex(w.index)
        for m in CAUSAL + NONCAUSAL:
            if m not in w or w[m].isna().all():
                continue
            o = w[m].astype(bool)
            a = int((w.acbfi_v3 & ~o).sum()); b = int((~w.acbfi_v3 & o).sum())
            r[f"mc_{m}"] = [a, b, f3(binomtest(a, a + b).pvalue) if a + b else None]
        v = sub[(sub.method == "acbfi_v3") & sub.success]
        r["step3_share"] = f3(v.stage.isin(["pair", "triple"]).mean())
        rows.append(r)
    A["main"] = rows
    tot = {"n_queries": int((e.method == "acbfi_v3").sum())}
    for m in meth:
        s = e[e.method == m]
        tot[f"succ_{m}"] = f3(s.success.mean()); tot[f"evals_{m}"] = f3(s.evals_total.mean())
        tot[f"Lact_{m}"] = f3(s[s.success].L_active.mean())
    v = e[(e.method == "acbfi_v3")]
    tot["step3_abs"] = f3((v.success & v.stage.isin(["pair", "triple"])).mean())
    tot["single_share_of_solved"] = f3(1 - tot["step3_abs"] / tot["succ_acbfi_v3"])
    R = {r["dataset"]: r for r in rows}
    tot["cheaper_than_untargeted"] = sorted(k for k, r in R.items() if r["evals_acbfi_v3"] < r["evals_untargeted"])
    tot["costlier_than_untargeted"] = sorted(k for k, r in R.items() if r["evals_acbfi_v3"] >= r["evals_untargeted"])
    tot["cheaper_than_v1_all"] = all(r["evals_acbfi_v3"] < r["evals_acbfi_v1"] for r in R.values())
    tot["cheaper_than_lewis_all"] = all(r["evals_acbfi_v3"] < r["evals_lewis_style"] for r in R.values())
    tot["shap_lower"] = sorted(k for k, r in R.items() if r["succ_shap_staged"] < r["succ_acbfi_v3"] - 0.005)
    # claim verdicts (the manuscript build checks its comparative sentences against these)

    def sig(r, m, worse):
        mc = r.get(f"mc_{m}")
        if not mc or mc[2] is None or mc[2] >= 0.05:
            return False
        return mc[1] > mc[0] if worse else mc[0] > mc[1]
    tot["sig_worse_causal"] = sorted(f"{k}|{m}" for k, r in R.items() for m in CAUSAL if sig(r, m, True))
    tot["sig_worse_noncausal"] = sorted(f"{k}|{m}" for k, r in R.items() for m in NONCAUSAL if sig(r, m, True))
    tot["v1_costlier_than_untargeted_all"] = all(r["evals_acbfi_v1"] > r["evals_untargeted"] for r in R.values())
    tot["Lact_fewest"] = all(tot["Lact_acbfi_v3"] < tot[f"Lact_{m}"] for m in meth if m != "acbfi_v3")
    tot["succ_dice"] = f3(dice.success.mean())
    A["main_total"] = tot
    # paired (both-success) costs from E0m_summarize
    pc = pd.read_csv(os.path.join(RES, "E0m", "E0m_paired_cost.csv"))
    out = {}
    for d in DS:
        s = pc[pc.dataset == d]
        rec = {}
        for m in ["untargeted", "lewis_style", "acbfi_v1"]:
            for met in ["cost", "cost_action"]:
                q = s[(s.method == m) & (s.metric == met)]
                if len(q):
                    q = q.iloc[0]
                    rec[f"{m}|{met}"] = [f3(q.acbfi_v3), f3(q.other), f3(q.p_holm), int(q.n_both)]
        out[NAMES[d]] = rec
    A["paired"] = out
    lw = [k for k, r in out.items() if "lewis_style|cost" in r and r["lewis_style|cost"][0] < r["lewis_style|cost"][1]]
    A["main_total"]["cost_below_lewis"] = len(lw)
    ua = [k for k, r in out.items() if "untargeted|cost_action" in r and r["untargeted|cost_action"][2] is not None
          and r["untargeted|cost_action"][2] < 0.05 and r["untargeted|cost_action"][0] > r["untargeted|cost_action"][1]]
    A["main_total"]["costact_higher_sig"] = sorted(ua)


# ---------------------------------------------------------------- synthetic --------------
def synthetic():
    s1 = pd.read_csv(os.path.join(RES, "synthetic", "S1_interaction_sources.csv"))
    A["S1"] = s1.to_dict(orient="records")
    s2 = pd.read_csv(os.path.join(RES, "synthetic", "S2_planted_synergy_lever.csv"))
    ok = s2[s2.oracle_success]
    t = ok[ok.group == "hidden_hard"].groupby(["model", "k", "rule"]).success.mean().unstack("rule")
    A["S2"] = {f"{m}|k={k}": {r: f3(v) for r, v in row.items()} for (m, k), row in t.iterrows()}
    s2c = pd.read_csv(os.path.join(RES, "synthetic", "S2_planted_synergy_cause.csv"))
    okc = s2c[s2c.oracle_success]
    tc = okc[okc.group == "hidden_hard"].groupby(["model", "k", "rule"]).success.mean().unstack("rule")
    A["S2_cause"] = {f"{m}|k={k}": {r: f3(v) for r, v in row.items()} for (m, k), row in tc.iterrows()}
    # figure 2
    rules = ["shap_abs", "cg1", "acbfi_node", "br_single", "br_pair"]
    labels = ["SHAP", "Interv. main\n(mean C_G1)", "Mean-based\nC_G1+C_G4", "Best-resp.\nsingle", "Best-resp.\npair (C_G4)"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (tab, title) in zip(axes, [(t, "Lever ≠ cause (immutable cause)"),
                                        (tc, "Lever = cause (control)")]):
        for i, (m, k) in enumerate([("true", 2), ("xgb", 2)]):
            vals = [tab.loc[(m, k), r] for r in rules]
            ax.bar(np.arange(len(rules)) + (i - 0.5) * 0.38, vals, width=0.38,
                   label="true model" if m == "true" else "XGBoost", color=["#4C72B0", "#DD8452"][i])
        ax.set_xticks(range(len(rules))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("Recourse success (k = 2)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_synthetic.png"), dpi=300); plt.close(fig)


# ---------------------------------------------------------------- E6 ---------------------
def escalation():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E6", "E6_*_seed*.csv"))
                    if os.path.getsize(f) > 10], ignore_index=True)
    rules = ["br_cg4", "mean_cg4", "additive", "shap", "random", "topM_all", "all_pairs"]
    tot = df.groupby("rule").agg(success=("success", "mean"), evals=("evals", "mean"))
    w = df.pivot_table(index=["dataset", "seed", "model", "query"], columns="rule",
                       values="success", aggfunc="first").astype(bool)
    out = {"n": int(len(w)), "by_dataset": df.groupby("dataset").query.size().div(len(rules)).astype(int).to_dict(),
           "rules": {}}
    for r in rules:
        rec = {"success": f3(tot.loc[r, "success"]), "evals": f3(tot.loc[r, "evals"])}
        if r != "br_cg4":
            a = int((w.br_cg4 & ~w[r]).sum()); b = int((~w.br_cg4 & w[r]).sum())
            rec["mc"] = [a, b, float(binomtest(a, a + b).pvalue) if a + b else None]
        out["rules"][r] = rec
    A["E6"] = out


# ---------------------------------------------------------------- E7 ---------------------
def difficulty():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E7", "E7_*_seed*.csv"))],
                   ignore_index=True)
    A["E7_seeds"] = sorted(int(s) for s in df.seed.unique())
    A["E7_seeds_by_dataset"] = df.groupby("dataset").seed.nunique().to_dict()
    w = df.pivot_table(index=["dataset", "act_frac", "tau", "seed", "model", "query"],
                       columns="method", values="success", aggfunc="first").astype(bool)
    rows = []
    for (d, fr, tau), t in w.groupby(level=["dataset", "act_frac", "tau"]):
        a = int((t.acbfi_v3 & ~t.untargeted).sum()); b = int((~t.acbfi_v3 & t.untargeted).sum())
        rows.append({"dataset": d, "act_frac": fr, "tau": tau,
                     "v3": t.acbfi_v3.mean(), "topM": t.v3_topM.mean(), "single": t.v3_single.mean(),
                     "unt": t.untargeted.mean(), "a": a, "b": b,
                     "p": binomtest(a, a + b).pvalue if a + b else np.nan})
    s = pd.DataFrame(rows)
    s["gain"] = s.v3 - s.unt; s["esc"] = s.v3 - s.single; s["topM_gain"] = s.topM - s.v3
    ev = df.groupby(["act_frac", "tau", "method"]).evals.mean().unstack("method")
    A["E7"] = {
        "n_cells": int(len(s)),
        "sig_better": int(((s.p < .05) & (s.a > s.b)).sum()),
        "sig_worse": int(((s.p < .05) & (s.a < s.b)).sum()),
        "neg_cells": int((s.gain < 0).sum()), "min_gain": f3(s.gain.min()),
        "sig_datasets": sorted({NAMES[d] for d in s[(s.p < .05) & (s.a > s.b)].dataset}),
        "mean_gain": {f"{fr}|{tau}": f3(g) for (fr, tau), g in s.groupby(["act_frac", "tau"]).gain.mean().items()},
        "mean_esc": {f"{fr}|{tau}": f3(g) for (fr, tau), g in s.groupby(["act_frac", "tau"]).esc.mean().items()},
        "mean_topM_gain": {f"{fr}|{tau}": f3(g) for (fr, tau), g in s.groupby(["act_frac", "tau"]).topM_gain.mean().items()},
        "evals": {f"{fr}|{tau}": {m: f3(v) for m, v in row.items()} for (fr, tau), row in ev.iterrows()},
        "grow_all": int((s[s.act_frac == "all"].pivot(index="dataset", columns="tau", values="gain")
                         .pipe(lambda g: g[0.2] > g[0.5])).sum()),
        "grow_half": int((s[s.act_frac == "half"].pivot(index="dataset", columns="tau", values="gain")
                          .pipe(lambda g: g[0.2] > g[0.5])).sum()),
        "max_esc": s.sort_values("esc", ascending=False).head(3)[["dataset", "act_frac", "tau", "esc"]]
        .assign(dataset=lambda x: x.dataset.map(NAMES)).round(3).to_dict(orient="records"),
        "no_sig_datasets": sorted(NAMES[d] for d in set(s.dataset) - set(s[(s.p < .05)].dataset)),
    }
    ev2 = df[df.tau == 0.2].groupby(["dataset", "act_frac", "method"]).evals.mean().unstack("method")
    ev2["ratio"] = ev2["v3_topM"] / ev2["acbfi_v3"]
    top = ev2.sort_values("ratio", ascending=False).iloc[0]
    A["E7"]["max_topM_ratio"] = {"ratio": f3(top["ratio"]), "dataset": NAMES[top.name[0]],
                                 "act_frac": top.name[1], "v3": f3(top["acbfi_v3"]), "topM": f3(top["v3_topM"])}
    # figure 3
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    taus = [0.5, 0.4, 0.3, 0.2]
    for ax, fr in zip(axes, ["all", "half"]):
        g = s[s.act_frac == fr].pivot(index="dataset", columns="tau", values="gain")[taus]
        for d, row in g.iterrows():
            ax.plot(range(4), 100 * row.values, color="0.75", lw=1)
            if abs(row.values).max() > 0.04:
                ax.annotate(NAMES[d], (3, 100 * row.values[-1]), fontsize=6, xytext=(3, 0),
                            textcoords="offset points", va="center")
        ax.plot(range(4), 100 * g.mean().values, color="#C44E52", lw=2.5, marker="o", label="mean (10 datasets)")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"τ={t}" for t in taus])
        ax.set_title("All actionable features" if fr == "all" else "Random half of actionable features", fontsize=9)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Success gain of A-CBFI over\nuntargeted causal search (%p)")
    axes[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_difficulty.png"), dpi=300); plt.close(fig)


# ---------------------------------------------------------------- E4 ---------------------
def dag_sensitivity():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E4", "E4_*_seed*.csv"))],
                   ignore_index=True)
    df["key"] = df.dataset + "|" + df.model + "|" + df["query"].astype(str) + "|" + df.method
    ref = df[df.variant == "llm"].set_index("key")["actions"]
    df["same"] = np.where(df.success, df.actions == df.key.map(ref), np.nan)
    g = df.groupby(["dataset", "variant", "method"]).agg(success=("success", "mean"),
                                                          same=("same", "mean"),
                                                          valid_llm=("valid_under_llm", "mean"),
                                                          valid_np=("valid_no_prop", "mean"))
    A["E4"] = {f"{d}|{v}|{m}": {k: f3(x) for k, x in row.items()} for (d, v, m), row in g.iterrows()}
    w = df.pivot_table(index=["dataset", "variant", "model", "query"], columns="method",
                       values="success", aggfunc="first").astype(bool)
    neverworse = all(((t.acbfi_v3.mean() >= t.untargeted.mean())
                      for _, t in w.groupby(level=["dataset", "variant"])))
    A["E4_neverworse"] = bool(neverworse)
    st = g.reset_index()
    st = st[st.variant != "llm"].pivot_table(index="dataset", columns="method", values="same")
    A["E4_stability_mean"] = {NAMES[d]: {m: f3(v) for m, v in r.items()} for d, r in st.iterrows()}
    # figure 4
    vs = ["drop30_r1", "drop30_r2", "rev30_r1", "rev30_r2", "empty", "pc", "gbr"]
    dsl = [d for d in DS if d in st.index]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(dsl))
    for i, m in enumerate(["acbfi_v3", "untargeted"]):
        vals = [g.loc[(d, slice(None), m), "same"].reset_index().query("variant in @vs")["same"].mean() for d in dsl]
        ax.bar(x + (i - .5) * .38, vals, .38, label="A-CBFI" if m == "acbfi_v3" else "Untargeted causal",
               color=["#4C72B0", "#8C8C8C"][i])
    ax.set_xticks(x); ax.set_xticklabels([NAMES[d] for d in dsl], fontsize=8)
    ax.set_ylabel("Prescription agreement with\nreference DAG (mean of 7 variants)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_dag.png"), dpi=300); plt.close(fig)


# ---------------------------------------------------------------- figure 1 ---------------
# Figure 1 is drawn in manuscript_assets/fig1_pipeline.pptx (make_fig1_pptx.py, editable) and
# exported to fig1_pipeline.png from PowerPoint; it is not generated here.


# ---------------------------------------------------------------- case studies -----------
def case_studies():
    from core import CountingModel, acbfi_v3, mad_of, untargeted
    from R1_lever_vs_cause import make_models
    from real_data import load_task
    out = {}
    for name, d, mname, row in [("heloc", "heloc", "RF", 1828), ("cdc", "cdc_diabetes", "RF", 48855)]:
        t = load_task(d, 0)
        rng = np.random.default_rng(0)
        bg = t.X_train[rng.choice(len(t.X_train), min(2000, len(t.X_train)), replace=False)]
        clf = make_models()[mname].fit(t.X_train, t.y_train)
        fn = lambda X, c=clf: c.predict_proba(np.asarray(X))[:, 1]
        m = CountingModel(fn)
        x = t.X_test[row]
        r = acbfi_v3(m, t.scm, x, bg, t.actionable, rng=np.random.default_rng(1))
        u = untargeted(CountingModel(fn), t.scm, x, bg, t.actionable)
        raw = lambda v: v * t.sd + t.mu
        xr, cr = raw(x), raw(r.x_cf)
        changed = [j for j in range(len(x)) if abs(r.x_cf[j] - x[j]) > 1e-6]
        r2 = lambda v: float(round(v, 2))
        out[name] = {"dataset": NAMES[d], "model": mname, "p_before": f3(fn(x[None])[0]),
                     "p_after": f3(fn(r.x_cf[None])[0]), "stage": r.stage,
                     "untargeted_success": bool(u.success), "evals": int(r.evals_diag + r.evals_search),
                     "direct": [{"feature": t.features[j], "from": r2(xr[j]), "to": r2(cr[j])} for j in r.intervened],
                     "downstream": [{"feature": t.features[j], "from": r2(xr[j]), "to": r2(cr[j])}
                                    for j in changed if j not in r.intervened],
                     "top_F1": sorted(((t.features[k], f3(v)) for k, v in r.extra["F1"].items()),
                                      key=lambda z: -z[1])[:4]}
        # menu of successful single actions (Step 1 grid), cheapest first
        from core import grid_v1
        mad = mad_of(bg)
        menu = []
        for j in t.actionable:
            g = grid_v1(bg, j, 6, x, t.scm)
            if len(g) == 0:
                continue
            V = np.zeros((len(g), len(x))); Mk = np.zeros_like(V, bool)
            V[:, j] = g; Mk[:, j] = True
            Xcf = t.scm.do(x, V, Mk); p = fn(Xcf)
            ok = p < 0.5
            if ok.any():
                c = (np.abs(Xcf - x) / mad).sum(axis=1); c[~ok] = np.inf
                b = int(np.argmin(c))
                menu.append({"feature": t.features[j], "from": r2(raw(x)[j]), "to": r2(raw(Xcf[b])[j]),
                             "cost": f3(c[b]), "p_after": f3(p[b])})
        out[name]["menu"] = sorted(menu, key=lambda m: m["cost"])
    A["cases"] = out


def cg3_cg4():
    from scipy.stats import binomtest
    a = pd.read_csv(os.path.join(RES, "E8", "E8a_summary.csv"))
    A["E8a"] = a.to_dict(orient="records")
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E8", "E8_*_seed*.csv"))
                    if os.path.getsize(f) > 10], ignore_index=True)
    hard = df[df.kind == "hard"]
    tot = hard.groupby("rule").agg(success=("success", "mean"), evals=("evals", "mean"))
    w = hard.pivot_table(index=["dataset", "seed", "model", "query"], columns="rule",
                         values="success", aggfunc="first").astype(bool)
    mc = {}
    for r in ("br_cg4", "additive", "topM_all"):
        x = int((w.pg3 & ~w[r]).sum()); y = int((~w.pg3 & w[r]).sum())
        mc[r] = [x, y, float(binomtest(x, x + y).pvalue) if x + y else None]
    single = df[df.kind == "single"]
    by = single.groupby("dataset").agg(n=("n_alternatives", "size"),
                                       ge2=("n_alternatives", lambda v: (v >= 2).mean()),
                                       med=("n_alternatives", "median"),
                                       prem=("cost_second", lambda v: np.nan))
    prem = (single.cost_second / single.cost_best)
    by["prem"] = prem.groupby(single.dataset).median()
    summ = json.load(open(os.path.join(RES, "E8", "E8cd_summary.json")))
    A["E8"] = {
        "n_hard": int(len(w)), "n_single": int(len(single)),
        "hard": {r: {"success": f3(tot.loc[r, "success"]), "evals": f3(tot.loc[r, "evals"])} for r in tot.index},
        "mc": mc,
        "share_ge2": f3((single.n_alternatives >= 2).mean()),
        "median_alt": float(single.n_alternatives.median()),
        "median_premium": f3(prem.median()),
        "by_dataset": {NAMES[d]: {"n": int(r.n), "ge2": f3(r.ge2), "med": float(r.med), "prem": f3(r.prem)}
                       for d, r in by.iterrows()},
        "auc_pg4": f3(summ["E8c_auc_pg4"]), "E8d": {k: (f3(v) if isinstance(v, float) else v)
                                                   for k, v in summ["E8d"].items()},
    }


if __name__ == "__main__":
    datasets_table(); main_benchmark(); synthetic(); escalation(); difficulty()
    dag_sensitivity(); case_studies(); cg3_cg4()
    with open(os.path.join(OUT, "assets.json"), "w", encoding="utf-8") as f:
        json.dump(A, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("assets written:", sorted(A))
