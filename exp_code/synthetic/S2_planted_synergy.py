"""
S2. Planted-synergy SCM: when does additive targeting fail?

Nodes (topological): 0 immutable root (age-like); 1, 2 synergy pair (roots);
3, 4 decoys caused by 0; 5 decoy caused by 3; 6, 7 decoy roots.
Adverse logit = c - beta * x1 * x2 + sum_d 0.6 * tanh(x_d) + 0.5 * x0
Decoys have visible main effects but bounded reach; the pair (1, 2) has no main effect
at x1 = x2 = 0 but can cross the boundary jointly.

Query groups: 'hidden'      = adverse and |x1|, |x2| < 0.3 (synergy invisible to additive scores)
              'hidden_hard' = hidden and p_adv > 0.85 (far from the boundary)
              'hard'        = p_adv > 0.85
              'random'      = adverse, no constraint
Models: the true function, and XGBoost trained on Bernoulli labels from the SCM.
Every rule selects k targets and is searched with the SAME exhaustive grid search.
Output: exp_result/synthetic/S2_planted_synergy.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from synth_lib import SCM, evaluate_instance, sigmoid  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "exp_result", "synthetic")
os.makedirs(OUT, exist_ok=True)

N_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
# Setting 'lever': adverse risk is driven mainly by the immutable root (AGE_W large) and the
# actionable decoys have small reach, so recourse must come from a lever that is not a cause.
# Setting 'cause': decoys are strong (the earlier run), so causes and levers coincide.
SETTING = sys.argv[2] if len(sys.argv) > 2 else "lever"
DECOY_W, AGE_W = {"lever": (0.4, 1.5), "cause": (0.6, 0.5)}[SETTING]


def make_scm():
    parents = {3: [0], 4: [0], 5: [3]}
    eqs = {3: lambda X: 0.7 * X[:, 0], 4: lambda X: -0.5 * X[:, 0], 5: lambda X: 0.6 * X[:, 3]}
    return SCM(8, parents, eqs)


def make_logit(beta, c):
    def logit(X):
        return c - beta * X[:, 1] * X[:, 2] + DECOY_W * np.tanh(X[:, 3:8]).sum(axis=1) + AGE_W * X[:, 0]
    return logit


def main():
    rng = np.random.default_rng(1)
    scm = make_scm()
    actionable = list(range(1, 8))
    rows = []
    for beta in [0.5, 1.0, 2.0]:
        pop = scm.sample(20000, rng)
        raw = make_logit(beta, 0.0)(pop)
        c = -np.quantile(raw, 0.70)            # ~30 % adverse
        true_model = lambda X, f=make_logit(beta, c): sigmoid(f(X))
        models = {"true": true_model}
        try:
            from xgboost import XGBClassifier
            y = rng.random(len(pop)) < true_model(pop)
            xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                eval_metric="logloss", random_state=0).fit(pop, y)
            models["xgb"] = lambda X, m=xgb: m.predict_proba(X)[:, 1]
        except ImportError:
            pass
        bg = pop[:5000]
        for mname, model in models.items():
            p = model(pop)
            adverse = p > 0.5
            hidden = adverse & (np.abs(pop[:, 1]) < 0.3) & (np.abs(pop[:, 2]) < 0.3)
            hard = p > 0.85
            groups = (("hidden", hidden), ("hidden_hard", hidden & hard),
                      ("hard", hard), ("random", adverse))
            for group, mask in groups:
                if mask.sum() == 0:
                    continue
                idx = rng.choice(np.where(mask)[0], size=min(N_QUERIES, mask.sum()), replace=False)
                for q, i in enumerate(idx):
                    for r in evaluate_instance(model, scm, pop[i], bg, actionable, rng=rng):
                        rows.append({"beta": beta, "model": mname, "group": group, "query": q, **r})
                print(f"beta={beta} model={mname} group={group} done ({len(idx)} queries)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"S2_planted_synergy_{SETTING}.csv"), index=False, encoding="utf-8-sig")

    summ = (df.groupby(["model", "group", "beta", "k", "rule"])
              .agg(success=("success", "mean"), oracle=("oracle_success", "mean"),
                   cost=("cost", "mean"))
              .reset_index())
    piv = summ.pivot_table(index=["model", "group", "beta", "k"], columns="rule", values="success")
    piv["oracle"] = summ.groupby(["model", "group", "beta", "k"])["oracle"].first()
    piv.round(2).to_csv(os.path.join(OUT, f"S2_summary_success_{SETTING}.csv"), encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
