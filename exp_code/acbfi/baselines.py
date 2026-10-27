"""
Baselines for the revised benchmark.

random_grid_ce   the manuscript's "Wachter's CE": random sampling of up to 500 grid
                 combinations over all mutable features, no SCM (renamed; it is not Wachter's
                 gradient objective)
shap_targeted_ce the manuscript's "SHAP-Targeted CE" with REAL SHAP (the original used
                 model.feature_importances_ / uniform weights): top max(3, 0.25 m) features by
                 |SHAP|, then random_grid_ce restricted to them; no SCM
lewis_style      LEWIS-style recourse (Galhotra, Pradhan & Salimi, SIGMOD 2021), re-implemented
                 on the same SCM: sufficiency of an action = share of context neighbours (rows
                 closest to x on the immutable attributes, currently adverse) whose abducted
                 counterfactual under the action becomes favourable; returns the cheapest
                 action (for x) whose sufficiency >= alpha and that flips x; singles first,
                 then pairs among the attributes with the highest single sufficiency.
                 NOTE: a faithful-in-spirit re-implementation, not the authors' code.
"""

import time
from itertools import combinations, product

import numpy as np

from core import THRESHOLD, CountingModel, RecourseResult, _finish, grid_v1, mad_of


def random_grid_ce(model: CountingModel, x, bg, mutable, max_iter=500, seed=42, scm=None, **_):
    """scm: optional ConstrainedSCM whose direction/bound rules are applied to the candidate
    values (the SCM equations themselves are NOT used: this baseline is non-causal)."""
    t0 = time.time(); model.reset(); model.phase = "search"
    mad = mad_of(bg)
    n_per = max(5, int(np.power(max_iter, 1.0 / max(1, len(mutable)))))
    grids = {c: np.linspace(bg[:, c].min(), bg[:, c].max(), n_per) for c in mutable}
    total = n_per ** len(mutable)
    rng = np.random.default_rng(seed)
    if total > max_iter:
        combos = np.array([[rng.choice(grids[c]) for c in mutable] for _ in range(max_iter)])
    else:
        combos = np.array(list(product(*[grids[c] for c in mutable])))
    Xcf = np.tile(x, (len(combos), 1)).astype(float)
    Xcf[:, mutable] = combos
    if hasattr(scm, "clip"):
        Xcf = scm.clip(x, Xcf)
    ok = model(Xcf) < THRESHOLD
    res = RecourseResult()
    if ok.any():
        cost = (np.abs(Xcf - x) / mad).sum(axis=1)
        cost[~ok] = np.inf
        b = int(np.argmin(cost))
        res.success, res.x_cf = True, Xcf[b]
        res.intervened = [int(j) for j in np.where(np.abs(Xcf[b] - x) > 1e-9)[0]]
        res.stage = "grid"
    return _finish(res, model, t0)


def shap_targeted_ce(model: CountingModel, x, bg, mutable, phi, shap_evals=0, scm=None, **_):
    k = max(3, int(len(mutable) * 0.25))
    top = sorted(mutable, key=lambda j: -abs(phi[j]))[:k]
    res = random_grid_ce(model, x, bg, top, scm=scm)
    res.evals_diag += shap_evals
    return res


def lewis_style(model: CountingModel, scm, x, bg, act, immutable, alpha=0.5, n_ctx=50,
                bins=6, M=6, **_):
    t0 = time.time(); model.reset()
    mad = mad_of(bg)
    model.phase = "diagnosis"
    # context: neighbours on the immutable attributes (all attributes if none), adverse now
    key = immutable if len(immutable) else list(range(len(x)))
    dist = (np.abs(bg[:, key] - x[key]) / mad[key]).sum(axis=1)
    pool = bg[np.argsort(dist)[: 4 * n_ctx]]
    adverse = pool[model(pool) >= THRESHOLD]
    ctx = (adverse if len(adverse) >= 10 else pool)[:n_ctx]

    def sufficiency(targets, values):
        """values: (n_actions, len(targets)); returns sufficiency per action."""
        n, c = len(values), len(ctx)
        out = np.zeros(n)
        for k in range(n):
            V = np.zeros((c, len(x))); Mk = np.zeros_like(V, bool)
            for t, v in zip(targets, values[k]):
                V[:, t] = v; Mk[:, t] = True
            Xc = np.vstack([scm.do(u, V[:1], Mk[:1]) for u in ctx])   # per-neighbour abduction
            out[k] = np.mean(model(Xc) < THRESHOLD)
        return out

    def try_actions(targets):
        grids = [grid_v1(bg, t, bins, x, scm) for t in targets]
        vals = np.array(list(product(*grids)))
        if len(vals) == 0:
            return None, np.zeros(0), vals
        model.phase = "diagnosis"
        suf = sufficiency(targets, vals)
        model.phase = "search"
        cand = vals[suf >= alpha]
        if len(cand) == 0:
            return None, suf, vals
        V = np.zeros((len(cand), len(x))); Mk = np.zeros_like(V, bool)
        for k, t in enumerate(targets):
            V[:, t] = cand[:, k]; Mk[:, t] = True
        Xcf = scm.do(x, V, Mk)
        ok = model(Xcf) < THRESHOLD
        if not ok.any():
            return None, suf, vals
        cost = (np.abs(Xcf - x) / mad).sum(axis=1); cost[~ok] = np.inf
        b = int(np.argmin(cost))
        return (float(cost[b]), Xcf[b], list(targets)), suf, vals

    res = RecourseResult()
    best, single_suf = None, {}
    for t in act:
        r, suf, _ = try_actions([t])
        single_suf[t] = float(suf.max()) if len(suf) else 0.0
        if r and (best is None or r[0] < best[0]):
            best = r
    stage = "single"
    if best is None:
        top = sorted(act, key=lambda t: -single_suf[t])[:M]
        for p in combinations(top, 2):
            r, _, _ = try_actions(list(p))
            if r and (best is None or r[0] < best[0]):
                best = r
        stage = "pair"
    if best:
        res.success, res.x_cf, res.intervened, res.stage = True, best[1], best[2], stage
    res.extra = {"n_ctx": len(ctx)}
    return _finish(res, model, t0)
