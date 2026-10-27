"""
A-CBFI core (revision). Every method receives a CountingModel so that diagnosis and
search model evaluations are counted end-to-end (review Major 2).

Methods (all return a RecourseResult):
  acbfi_v2        revised A-CBFI: prescriptive (best-response) C_G1 diagnosis, staged search
                  single -> pair -> triple; pairwise best-response C_G4 is computed lazily,
                  only when no single lever reaches the boundary
  acbfi_v1        manuscript A-CBFI ported faithfully: mean-based |C_G1| + |C_G4| node score,
                  25th-percentile pruning (cap 15), interaction-guided beam search (IGBS)
  untargeted      the manuscript's "Untargeted Causal" baseline: same beam search over all
                  mutable features, no diagnosis (column order)
  staged_by_score acbfi_v2's staged search driven by an external ranking (e.g. |SHAP|):
                  the controlled ablation "diagnosis signal replaced, search unchanged"
"""

import os
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synthetic"))
from synth_lib import SCM  # noqa: E402

THRESHOLD = 0.5


# =========================================================================================
# Counting wrapper
# =========================================================================================

class CountingModel:
    """Wraps p_adv(X) and counts evaluated rows per phase ('diagnosis' / 'search')."""

    def __init__(self, fn):
        self.fn = fn
        self.counts: Dict[str, int] = {}
        self.phase = "search"

    def __call__(self, X):
        X = np.atleast_2d(X)
        self.counts[self.phase] = self.counts.get(self.phase, 0) + len(X)
        return self.fn(X)

    def reset(self):
        self.counts = {}
        self.phase = "search"


@dataclass
class RecourseResult:
    success: bool = False
    x_cf: Optional[np.ndarray] = None
    intervened: List[int] = field(default_factory=list)   # directly manipulated features
    stage: str = ""
    evals_diag: int = 0
    evals_search: int = 0
    time: float = 0.0
    extra: dict = field(default_factory=dict)


def mad_of(bg):
    mad = np.median(np.abs(bg - np.median(bg, axis=0)), axis=0)
    mad[mad < 1e-6] = 1.0
    return mad


def grid_v1(bg, j, bins=6, x=None, scm=None):
    """Manuscript discretisation: unique values if few, else quantiles at linspace(0, 1, bins).
    With x and a constrained SCM, only values in the allowed direction / bounds are kept."""
    col = bg[:, j]
    u = np.unique(col)
    g = u if len(u) <= bins else np.unique(np.quantile(col, np.linspace(0, 1, bins)))
    if x is not None and hasattr(scm, "allowed"):
        g = scm.allowed(x, j, g)
    return g


def _finish(res, model, t0):
    res.evals_diag = model.counts.get("diagnosis", 0)
    res.evals_search = model.counts.get("search", 0)
    res.time = time.time() - t0
    return res


def _search_targets(model, scm, x, targets, bg, mad, bins):
    """Grid over do() values of `targets` (each may stay untouched); min-cost success."""
    from itertools import product
    options = [np.concatenate([[np.nan], grid_v1(bg, t, bins, x, scm)]) for t in targets]
    combos = np.array(list(product(*options)))
    combos = combos[~np.all(np.isnan(combos), axis=1)]
    if len(combos) == 0:
        return None
    d = len(x)
    V = np.zeros((len(combos), d)); M = np.zeros((len(combos), d), bool)
    for k, t in enumerate(targets):
        M[:, t] = ~np.isnan(combos[:, k]); V[:, t] = np.nan_to_num(combos[:, k])
    Xcf = scm.do(x, V, M)
    ok = model(Xcf) < THRESHOLD
    if not ok.any():
        return None
    cost = (np.abs(Xcf - x) / mad).sum(axis=1)
    cost[~ok] = np.inf
    b = int(np.argmin(cost))
    return float(cost[b]), Xcf[b], [t for k, t in enumerate(targets) if M[b, t]]


# =========================================================================================
# Revised A-CBFI (v2)
# =========================================================================================

def _best_response_single(model, scm, x, rows, act):
    p0 = model(x[None])[0]
    F1 = {}
    for i in act:
        Mk = np.zeros(rows.shape, bool); Mk[:, i] = True
        F1[i] = max(0.0, float((p0 - model(scm.do(x, rows, Mk))).max()))
    return F1


def _best_response_pairs(model, scm, x, rows, pairs):
    p0 = model(x[None])[0]
    F2 = {}
    for i, j in pairs:
        Mk = np.zeros(rows.shape, bool); Mk[:, [i, j]] = True
        F2[(i, j)] = max(0.0, float((p0 - model(scm.do(x, rows, Mk))).max()))
    return F2


def staged_search(model, scm, x, bg, act, ranking: Sequence[int], pair_score, bins=6,
                  L=3, M=6, P=3, T=2):
    """Stage 1: grid-search each of the top-L single levers; Stage 2 (only if stage 1 fails):
    top-P pairs among the top-M features by pair_score; Stage 3: top-T triples (best pair +
    best third). Returns (cost, x_cf, intervened, stage) or None."""
    mad = mad_of(bg)
    best = None
    for t in ranking[:L]:
        r = _search_targets(model, scm, x, [t], bg, mad, bins)
        if r and (best is None or r[0] < best[0]):
            best = r
    if best:
        return (*best, "single")
    top = list(ranking[:M])
    pairs = list(combinations(top, 2))
    ps = pair_score(pairs)
    ranked_pairs = sorted(pairs, key=lambda p: -ps[p])
    for p in ranked_pairs[:P]:
        r = _search_targets(model, scm, x, list(p), bg, mad, bins)
        if r and (best is None or r[0] < best[0]):
            best = r
    if best:
        return (*best, "pair")
    for p in ranked_pairs[:T]:
        rest = [t for t in top if t not in p]
        if not rest:
            continue
        third = max(rest, key=lambda t: max(ps.get(tuple(sorted((p[0], t), key=top.index)), 0),
                                            ps.get(tuple(sorted((p[1], t), key=top.index)), 0)))
        r = _search_targets(model, scm, x, list(p) + [third], bg, mad, bins)
        if r and (best is None or r[0] < best[0]):
            best = r
    if best:
        return (*best, "triple")
    return None


def acbfi_v2(model: CountingModel, scm: SCM, x, bg, act, n_samples=50, bins=6, rng=None, **kw):
    t0 = time.time(); model.reset()
    rng = rng or np.random.default_rng(0)
    rows = bg[rng.choice(len(bg), n_samples, replace=False)]
    model.phase = "diagnosis"
    F1 = _best_response_single(model, scm, x, rows, act)
    ranking = sorted(act, key=lambda i: -F1[i])

    def pair_score(pairs):        # lazily computed best-response joint gain (C_G4 stage)
        model.phase = "diagnosis"
        out = _best_response_pairs(model, scm, x, rows, pairs)
        model.phase = "search"
        return out

    model.phase = "search"
    r = staged_search(model, scm, x, bg, act, ranking, pair_score, bins=bins, **kw)
    res = RecourseResult(extra={"F1": F1})
    if r:
        res.success, res.x_cf, res.intervened, res.stage = True, r[1], r[2], r[3]
    return _finish(res, model, t0)


def acbfi_v3(model: CountingModel, scm: SCM, x, bg, act, bins=6, n_pair_samples=30, M=6, P=3,
             T=2, rng=None, escalate=True, **_):
    """Revised A-CBFI with a grid-probe diagnosis.
    Diagnosis = do(x_i = g) for every actionable i and every grid value g (SCM propagation):
    it yields the prescriptive single-lever score F1_i = max_g [p(x) - p(do(x_i=g))] AND the
    cheapest single-lever recourse at the same time. Only if no single lever crosses the
    boundary, pairwise best-response gains (C_G4 stage) are estimated on background rows for
    the top-M features and the top-P pairs (then top-T triples) are grid-searched."""
    t0 = time.time(); model.reset()
    rng = rng or np.random.default_rng(0)
    mad = mad_of(bg)
    p0 = None
    model.phase = "diagnosis"
    p0 = model(x[None])[0]
    F1, best = {}, None
    for i in act:
        g = grid_v1(bg, i, bins, x, scm)
        if len(g) == 0:
            F1[i] = 0.0
            continue
        V = np.zeros((len(g), len(x))); Mk = np.zeros_like(V, bool)
        V[:, i] = g; Mk[:, i] = True
        Xcf = scm.do(x, V, Mk)
        p = model(Xcf)
        F1[i] = max(0.0, float((p0 - p).max()))
        ok = p < THRESHOLD
        if ok.any():
            cost = (np.abs(Xcf - x) / mad).sum(axis=1); cost[~ok] = np.inf
            b = int(np.argmin(cost))
            if best is None or cost[b] < best[0]:
                best = (float(cost[b]), Xcf[b], [i])
    res = RecourseResult(extra={"F1": F1})
    if best:
        res.success, res.x_cf, res.intervened, res.stage = True, best[1], best[2], "single"
        return _finish(res, model, t0)
    if not escalate or len(act) < 2:
        return _finish(res, model, t0)
    ranking = sorted(act, key=lambda i: -F1[i])
    top = ranking[:M]
    if escalate == "topM":
        # E6 finding: under small grids, exhaustively searching all pairs (then triples)
        # among the top-M single levers is cheaper and stronger than C_G4 screening.
        model.phase = "search"
        for size, stage in ((2, "pair"), (3, "triple")):
            for s in combinations(top, size):
                r = _search_targets(model, scm, x, list(s), bg, mad, bins)
                if r and (best is None or r[0] < best[0]):
                    best = (*r, stage)
            if best:
                break
        if best:
            res.success, res.x_cf, res.intervened, res.stage = True, best[1], best[2], best[3]
        return _finish(res, model, t0)
    rows = bg[rng.choice(len(bg), n_pair_samples, replace=False)]

    def pair_score(pairs):
        model.phase = "diagnosis"
        out = _best_response_pairs(model, scm, x, rows, pairs)
        model.phase = "search"
        return out

    model.phase = "search"
    ps = pair_score(list(combinations(top, 2)))
    ranked = sorted(ps, key=lambda p: -ps[p])
    for p in ranked[:P]:
        r = _search_targets(model, scm, x, list(p), bg, mad, bins)
        if r and (best is None or r[0] < best[0]):
            best = (*r, "pair")
    if best is None:
        for p in ranked[:T]:
            rest = [t for t in top if t not in p]
            if not rest:
                continue
            third = max(rest, key=lambda t: max(ps.get((p[0], t), ps.get((t, p[0]), 0)),
                                                ps.get((p[1], t), ps.get((t, p[1]), 0))))
            r = _search_targets(model, scm, x, list(p) + [third], bg, mad, bins)
            if r and (best is None or r[0] < best[0]):
                best = (*r, "triple")
    if best:
        res.success, res.x_cf, res.intervened, res.stage = True, best[1], best[2], best[3]
    return _finish(res, model, t0)


def staged_by_score(model: CountingModel, scm, x, bg, act, score: Dict[int, float],
                    diag_evals=0, bins=6, **kw):
    """Ablation: same staged search, ranking and pair scores from an external attribution
    (pair score = sum of the two scores). diag_evals: evaluations spent computing `score`."""
    t0 = time.time(); model.reset()
    ranking = sorted(act, key=lambda i: -score[i])
    model.phase = "search"
    r = staged_search(model, scm, x, bg, act, ranking,
                      lambda pairs: {p: score[p[0]] + score[p[1]] for p in pairs}, bins=bins, **kw)
    res = RecourseResult()
    if r:
        res.success, res.x_cf, res.intervened, res.stage = True, r[1], r[2], r[3]
    res = _finish(res, model, t0)
    res.evals_diag += diag_evals
    return res


# =========================================================================================
# Manuscript beam search (IGBS) -- ported from the preliminary version (arXiv:2608.27821v1)
# =========================================================================================

def igbs_beam(model, scm, x, targets, bg, budget=500, beam_width=5, bins=6,
              self_scores=None, pair_matrix=None):
    mad = mad_of(bg)
    max_depth = min(len(targets), 5)
    grids = {t: grid_v1(bg, t, bins, x, scm) for t in targets}
    self_scores = self_scores or {t: 0.0 for t in targets}
    s_scale = max(self_scores.values(), default=0.0)
    p_scale = 0.0
    if pair_matrix is not None:
        p_scale = max((abs(pair_matrix[i, j]) for i, j in combinations(targets, 2)), default=0.0)

    def prio(c, active):
        s = self_scores.get(c, 0.0) / s_scale if s_scale > 0 else 0.0
        p = 0.0
        if active and pair_matrix is not None and p_scale > 0:
            p = max(abs(pair_matrix[a, c]) for a in active) / p_scale
        return 0.5 * s + 0.5 * p

    beams = [{"act": {}, "score": 0.0, "prio": 0.0}]
    done = []
    evals = 0
    for _ in range(max_depth):
        nxt = []
        for beam in beams:
            active = list(beam["act"])
            cands = sorted([t for t in targets if t not in active], key=lambda t: -prio(t, active))
            for t in cands:
                if evals >= budget:
                    break
                vals = grids[t][: budget - evals]
                if len(vals) == 0:          # no allowed value for this feature
                    continue
                evals += len(vals)
                V = np.zeros((len(vals), len(x))); Mk = np.zeros_like(V, bool)
                for a_, v_ in beam["act"].items():
                    V[:, a_] = v_; Mk[:, a_] = True
                V[:, t] = vals; Mk[:, t] = True
                Xcf = scm.do(x, V, Mk)
                p = model(Xcf)
                cost = (np.abs(Xcf - x) / mad).sum(axis=1)
                pr = prio(t, active)
                for k in range(len(vals)):
                    act = dict(beam["act"]); act[t] = float(vals[k])
                    if p[k] < THRESHOLD:
                        done.append((cost[k], Xcf[k], list(act)))
                    else:
                        nxt.append({"act": act, "score": cost[k] + 5.0 * -np.log(1 - p[k] + 1e-9),
                                    "prio": pr})
            if evals >= budget:
                break
        if evals >= budget or not nxt:
            break
        nxt.sort(key=lambda b: (b["score"], -b["prio"]))
        beams = nxt[:beam_width]
    if not done:
        return None
    c, xcf, act = min(done, key=lambda z: z[0])
    return c, xcf, act


def acbfi_v1(model: CountingModel, scm, x, bg, act, n_samples=50, bins=6, budget=500, rng=None):
    """Manuscript A-CBFI: mean-based diagnosis (C6 fixed: node C_G4 = mean over all actionable
    partners, no clipping), 25th-percentile pruning with cap 15, IGBS."""
    from synth_lib import diagnose
    t0 = time.time(); model.reset()
    rng = rng or np.random.default_rng(0)
    model.phase = "diagnosis"
    dg = diagnose(model, scm, x, bg, act, n_samples=n_samples, rng=rng)
    node4 = {i: float(np.mean([dg["cg4"][i, j] for j in act if j != i])) if len(act) > 1 else 0.0
             for i in act}
    total = {i: abs(dg["cg1"][i]) + abs(node4[i]) for i in act}
    thr = np.percentile(list(total.values()), 25)
    targets = sorted([i for i in act if total[i] >= thr], key=lambda i: -total[i])[:15]
    model.phase = "search"
    r = igbs_beam(model, scm, x, targets, bg, budget=budget, bins=bins,
                  self_scores={i: abs(node4[i]) for i in targets}, pair_matrix=dg["cg4"])
    res = RecourseResult(extra={"n_targets": len(targets)})
    if r:
        res.success, res.x_cf, res.intervened, res.stage = True, r[1], r[2], "beam"
    return _finish(res, model, t0)


def untargeted(model: CountingModel, scm, x, bg, act, bins=6, budget=500, **_):
    t0 = time.time(); model.reset(); model.phase = "search"
    r = igbs_beam(model, scm, x, list(act), bg, budget=budget, bins=bins)
    res = RecourseResult()
    if r:
        res.success, res.x_cf, res.intervened, res.stage = True, r[1], r[2], "beam"
    return _finish(res, model, t0)
