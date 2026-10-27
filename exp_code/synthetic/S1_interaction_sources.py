"""
S1. Where does interventional interaction come from?

Three 3-node toy SCMs on the logit scale (so closed forms are exact):
  (a) model interaction   : no edges,                 logit = beta * x1 * x2
  (b) path blocking       : x1 -> x2 (x2 = a x1 + u), logit = w1 x1 + w2 x2   (additive model!)
  (c) shared descendant   : x3 = gamma x1 x2 + u,     logit = w3 x3           (model ignores x1, x2)

For each we compare, in the gain convention g = f(x) - f(do(.)):
  - manuscript C_G4(1,2) (signed mean over background rows), estimated
  - closed form E[C_G4]
  - model-only interventional SHAP interaction index (sign-flipped to gain convention)
  - model-only SHAP values of x1, x2
Output: exp_result/synthetic/S1_interaction_sources.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from synth_lib import SCM, diagnose, exact_shap, exact_shap_interaction  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "exp_result", "synthetic")
os.makedirs(OUT, exist_ok=True)

beta, a, w1, w2, gamma, w3 = 2.0, 0.8, 1.0, 1.5, 1.0, 2.0
rng = np.random.default_rng(0)

cases = {
    "(a) model interaction": (
        SCM(3, {}, {}),
        lambda X: beta * X[:, 0] * X[:, 1],
        lambda x: -beta * x[0] * x[1],
    ),
    "(b) path blocking": (
        SCM(3, {1: [0]}, {1: lambda X: a * X[:, 0]}),
        lambda X: w1 * X[:, 0] + w2 * X[:, 1],
        lambda x: -w2 * a * x[0],
    ),
    "(c) shared descendant": (
        SCM(3, {2: [0, 1]}, {2: lambda X: gamma * X[:, 0] * X[:, 1]}),
        lambda X: w3 * X[:, 2],
        lambda x: -w3 * gamma * x[0] * x[1],
    ),
}

rows = []
for name, (scm, f, closed) in cases.items():
    bg = scm.sample(20000, rng)
    for x12 in [(0.0, 0.0), (1.0, 0.8), (-1.0, 0.8)]:
        # build a factual point that is consistent with the SCM noise structure
        x = np.array([x12[0], x12[1], 0.0])
        if name.startswith("(b)"):
            x[1] = x12[1]  # x2 given directly (noise absorbs the rest)
        if name.startswith("(c)"):
            x[2] = gamma * x[0] * x[1] + 0.3
        diag = diagnose(f, scm, x, bg, actionable=[0, 1], n_samples=20000, rng=rng)
        phi, v = exact_shap(f, x, bg[:2000])
        sii = exact_shap_interaction(v, 3, 0, 1)
        rows.append({
            "mechanism": name,
            "x1": x[0], "x2": x[1],
            "C_G4 manuscript (est.)": round(diag["cg4"][0, 1], 3),
            "C_G4 closed form": round(closed(x), 3),
            "C_G4 positive-part (est.)": round(diag["cg4_pos"][0, 1], 3),
            "best-response interaction (est.)": round(diag["br"][0, 1], 3),
            "SHAP interaction index (gain sign)": round(-sii, 3),
            "SHAP phi_x1": round(phi[0], 3),
            "SHAP phi_x2": round(phi[1], 3),
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "S1_interaction_sources.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(df.to_string(index=False))
