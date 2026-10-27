"""
Synthetic SCM toolkit for the A-CBFI originality feasibility study.

Everything here is vectorised over batches of interventions so that exact
(enumerated) quantities can be computed on small graphs:

- SCM with known structural equations, abduction and batched do()-propagation
- C_G1 / C_G4 interventional diagnosis (same definition as the manuscript:
  gain = p_adv(factual) - p_adv(do(.)), C_G4 = joint gain - sum of single gains,
  values drawn from the same background row, no clipping)
- exact interventional SHAP values and Shapley interaction index (model only,
  no SCM) by subset enumeration
- exhaustive grid recourse search restricted to a target set, with SCM propagation
"""

from itertools import combinations, product
from math import factorial
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# =========================================================================================
# 1. SCM
# =========================================================================================

class SCM:
    """Nodes are 0..d-1 in topological order. eqs[j] maps X (n,d) -> (n,) using parents only."""

    def __init__(self, d: int, parents: Dict[int, List[int]], eqs: Dict[int, Callable]):
        self.d = d
        self.parents = {j: list(parents.get(j, [])) for j in range(d)}
        self.eqs = eqs
        for j, ps in self.parents.items():
            assert all(p < j for p in ps), "nodes must be topologically ordered"

    def descendants(self, i: int) -> set:
        out, frontier = set(), {i}
        while frontier:
            nxt = {j for j in range(self.d) for p in self.parents[j] if p in frontier}
            nxt -= out
            out |= nxt
            frontier = nxt
        return out

    def sample(self, n: int, rng: np.random.Generator, noise_scale: float = 1.0) -> np.ndarray:
        X = np.zeros((n, self.d))
        U = rng.normal(0, noise_scale, size=(n, self.d))
        for j in range(self.d):
            X[:, j] = (self.eqs[j](X) if self.parents[j] else 0.0) + U[:, j]
        return X

    def abduct(self, x: np.ndarray) -> np.ndarray:
        X = x[None, :]
        u = x.copy()
        for j in range(self.d):
            if self.parents[j]:
                u[j] = x[j] - self.eqs[j](X)[0]
        return u

    def abduct_rows(self, X: np.ndarray) -> np.ndarray:
        """Noise of every row (roots: the value itself)."""
        U = X.astype(float).copy()
        for j in range(self.d):
            if self.parents[j]:
                U[:, j] = X[:, j] - self.eqs[j](X)
        return U

    def generate(self, U: np.ndarray, V: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Interventional sample: rows of noise U, nodes in M fixed to V (no abduction)."""
        X = np.zeros_like(U, dtype=float)
        for j in range(self.d):
            X[:, j] = (self.eqs[j](X) + U[:, j]) if self.parents[j] else U[:, j]
            X[:, j] = np.where(M[:, j], V[:, j], X[:, j])
        return X

    def do(self, x: np.ndarray, V: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Batched counterfactual: rows of M mark intervened nodes, V gives their values.
        Non-intervened nodes keep their abducted factual noise."""
        u = self.abduct(x)
        n = V.shape[0]
        X = np.tile(x, (n, 1)).astype(float)
        for j in range(self.d):
            if self.parents[j]:
                X[:, j] = self.eqs[j](X) + u[j]
            X[:, j] = np.where(M[:, j], V[:, j], X[:, j])
        return X


def empty_scm(d: int) -> SCM:
    return SCM(d, {}, {})


# =========================================================================================
# 2. Interventional diagnosis (C_G1, C_G4)
# =========================================================================================

def _gain(model, x, X_cf):
    p0 = model(x[None, :])[0]
    return p0 - model(X_cf)


def diagnose(model, scm: SCM, x: np.ndarray, background: np.ndarray, actionable: Sequence[int],
             n_samples: int = 50, rng: Optional[np.random.Generator] = None) -> dict:
    """Interventional diagnosis on actionable features, all from the SAME model evaluations:

    cg1[i]       = E_v[g_i(v)]                              (manuscript C_G1)
    cg4[i,j]     = E_v[g_ij - g_i - g_j]                     (manuscript C_G4, signed mean)
    cg4_pos[i,j] = E_v[max(0, g_ij - g_i - g_j)]             (positive-part variant)
    F1[i]        = max_v g_i(v)                             (best-response single gain)
    F2[i,j]      = max_v g_ij(v)                            (best-response joint gain)
    br[i,j]      = F2 - F1_i - F1_j                         (best-response interaction)
    where g = p_adv(factual) - p_adv(do(.)) and v are background rows."""
    rng = rng or np.random.default_rng(0)
    d = scm.d
    rows = background[rng.choice(len(background), size=n_samples, replace=False)]
    out = {k: np.zeros(d) for k in ("cg1", "F1")}
    out.update({k: np.zeros((d, d)) for k in ("cg4", "cg4_pos", "F2", "br")})
    single = {}
    evals = 0
    for i in actionable:
        M = np.zeros((n_samples, d), bool); M[:, i] = True
        g = _gain(model, x, scm.do(x, rows, M)); evals += n_samples
        single[i] = g
        out["cg1"][i] = g.mean()
        out["F1"][i] = max(0.0, g.max())
    for i, j in combinations(actionable, 2):
        M = np.zeros((n_samples, d), bool); M[:, [i, j]] = True
        gj = _gain(model, x, scm.do(x, rows, M)); evals += n_samples
        delta = gj - single[i] - single[j]
        f2 = max(0.0, gj.max(), out["F1"][i], out["F1"][j])
        for k, val in (("cg4", delta.mean()), ("cg4_pos", np.maximum(delta, 0).mean()),
                       ("F2", f2), ("br", f2 - out["F1"][i] - out["F1"][j])):
            out[k][i, j] = out[k][j, i] = val
    out["evals"] = evals
    return out


# =========================================================================================
# 3. Exact interventional SHAP (model only, observational background, no SCM)
# =========================================================================================

def _coalition_values(model, x, background):
    d = len(x)
    vals = {}
    for r in range(d + 1):
        for S in combinations(range(d), r):
            Xb = background.copy()
            if S:
                Xb[:, list(S)] = x[list(S)]
            vals[frozenset(S)] = model(Xb).mean()
    return vals


def exact_shap(model, x, background):
    d = len(x)
    v = _coalition_values(model, x, background)
    phi = np.zeros(d)
    for i in range(d):
        others = [k for k in range(d) if k != i]
        for r in range(d):
            w = factorial(r) * factorial(d - r - 1) / factorial(d)
            for S in combinations(others, r):
                S = frozenset(S)
                phi[i] += w * (v[S | {i}] - v[S])
    return phi, v


def exact_shap_interaction(v, d, i, j):
    """Shapley interaction index (Grabisch) for pair (i, j) from coalition values."""
    others = [k for k in range(d) if k not in (i, j)]
    tot = 0.0
    for r in range(d - 1):
        w = factorial(r) * factorial(d - r - 2) / factorial(d - 1)
        for S in combinations(others, r):
            S = frozenset(S)
            tot += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
    return tot


def causal_shap(model, scm: SCM, x, background, n_perm=64, n_bg=50, rng=None, players=None):
    """Causal Shapley values (Heskes et al. 2020) by permutation sampling:
    v(S) = E_U[ f(X) | do(X_S = x_S) ], U drawn from noises abducted from background rows.
    players: features that take part (default all); the others are always sampled."""
    rng = rng or np.random.default_rng(0)
    d = scm.d
    players = list(range(d)) if players is None else list(players)
    U = scm.abduct_rows(background[rng.choice(len(background), n_bg, replace=False)])
    phi = np.zeros(d)
    perms = [rng.permutation(players) for _ in range(n_perm)]
    # all prefix coalitions of all permutations in one batch
    masks = []
    for perm in perms:
        m = np.zeros(d, bool)
        masks.append(m.copy())
        for p in perm:
            m[p] = True
            masks.append(m.copy())
    masks = np.array(masks)                                   # (n_perm*(P+1), d)
    nC = len(masks)
    Mb = np.repeat(masks, n_bg, axis=0)
    Ub = np.tile(U, (nC, 1))
    Vb = np.tile(x, (nC * n_bg, 1))
    vals = model(scm.generate(Ub, Vb, Mb)).reshape(nC, n_bg).mean(axis=1)
    P = len(players)
    for r, perm in enumerate(perms):
        base = r * (P + 1)
        for t, p in enumerate(perm):
            phi[p] += vals[base + t + 1] - vals[base + t]
    return phi / n_perm


# =========================================================================================
# 4. Recourse search restricted to a target set
# =========================================================================================

def grid_for(background, j, bins=7):
    u = np.unique(background[:, j])
    if len(u) <= bins + 1:          # discrete / ordinal-encoded feature: use observed levels
        return u
    return np.unique(np.quantile(background[:, j], np.linspace(0.02, 0.98, bins)))


def recourse_on_targets(model, scm: SCM, x, targets: Sequence[int], background, mad,
                        threshold=0.5, bins=7, cost_cols=None):
    """Exhaustive grid search over do() values of the target set (each target may also be
    left untouched). Returns (success, min MAD-L1 cost, best X_cf, n_evals).
    cost_cols: columns counted in the cost (default all; exclude latent outcome nodes)."""
    d = scm.d
    options = [np.concatenate([[np.nan], grid_for(background, t, bins)]) for t in targets]
    combos = np.array(list(product(*options)))
    combos = combos[~np.all(np.isnan(combos), axis=1)]
    V = np.zeros((len(combos), d)); M = np.zeros((len(combos), d), bool)
    for k, t in enumerate(targets):
        M[:, t] = ~np.isnan(combos[:, k])
        V[:, t] = np.nan_to_num(combos[:, k])
    Xcf = scm.do(x, V, M)
    p = model(Xcf)
    ok = p < threshold
    if not ok.any():
        return False, np.nan, None, len(combos)
    cols = slice(None) if cost_cols is None else list(cost_cols)
    cost = (np.abs(Xcf - x)[:, cols] / mad[cols]).sum(axis=1)
    cost[~ok] = np.inf
    b = int(np.argmin(cost))
    return True, float(cost[b]), Xcf[b], len(combos)


# =========================================================================================
# 5. Target-selection rules (all return exactly k targets)
# =========================================================================================

def top_k(score: np.ndarray, actionable, k):
    a = list(actionable)
    return [a[i] for i in np.argsort(-np.asarray([score[j] for j in a]), kind="stable")[:k]]


def _node_marginal(mat, a):
    return np.array([mat[i, [j for j in a if j != i]].mean() if len(a) > 1 else 0.0
                     for i in range(mat.shape[0])])


def _pair_then_greedy(a, k, pair_score, add_score):
    best = max(combinations(a, 2), key=lambda p: pair_score(*p))
    sel = list(best)
    while len(sel) < k:
        rest = [j for j in a if j not in sel]
        sel.append(max(rest, key=lambda j: add_score(sel, j)))
    return sel[:k]


def select_targets(rule: str, k: int, actionable, phi=None, diag=None, rng=None):
    """Target-selection rules compared under the same search. All return exactly k targets."""
    a = list(actionable)
    if rule == "shap_abs":          # additive, model-only (SHAP-targeted CE)
        return top_k(np.abs(phi), a, k)
    if rule == "shap_pos":
        return top_k(phi, a, k)
    cg1, cg4, cg4p, F1, F2 = (diag[n] for n in ("cg1", "cg4", "cg4_pos", "F1", "F2"))
    if rule == "cg1":               # additive, interventional (C_G1 only)
        return top_k(cg1, a, k)
    if rule == "acbfi_node":        # manuscript Eq.(6)-(7): |C_G1| + |mean_j C_G4(i,j)|
        return top_k(np.abs(cg1) + np.abs(_node_marginal(cg4, a)), a, k)
    if rule == "acbfi_node_pos":    # same, positive-part C_G4
        return top_k(np.abs(cg1) + _node_marginal(cg4p, a), a, k)
    if rule == "br_single":         # additive best-response: top-k single max gains
        return top_k(F1, a, k)
    if rule == "br_pair":           # best-response pair (estimated F({i,j})), then greedy
        return _pair_then_greedy(a, k, lambda i, j: F2[i, j],
                                 lambda sel, j: max(F2[s, j] for s in sel))
    if rule == "random":
        return list((rng or np.random.default_rng(0)).choice(a, size=k, replace=False))
    raise ValueError(rule)


def oracle_targets(model, scm, x, actionable, k, background, mad, bins=7, cost_cols=None):
    best = (False, np.inf, None)
    for T in combinations(actionable, k):
        ok, c, _, _ = recourse_on_targets(model, scm, x, T, background, mad, bins=bins,
                                          cost_cols=cost_cols)
        if ok and c < best[1]:
            best = (True, c, list(T))
    return best


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


RULES = ["shap_abs", "cg1", "acbfi_node", "acbfi_node_pos", "br_single", "br_pair", "random"]


def evaluate_instance(model, scm, x, bg, actionable, ks=(2, 3), rules=RULES,
                      shap_bg=200, n_samples=50, rng=None, phi=None, bins=7):
    """Run every target-selection rule with the same exhaustive search and the oracle.
    Returns a list of dict rows (one per rule x k) plus instance-level interaction stats.
    phi: precomputed SHAP values (otherwise exact enumeration, only feasible for small d)."""
    rng = rng or np.random.default_rng(0)
    mad = np.median(np.abs(bg - np.median(bg, axis=0)), axis=0)
    mad[mad < 1e-6] = 1.0
    if phi is None:
        phi, _ = exact_shap(model, x, bg[rng.choice(len(bg), shap_bg, replace=False)])
    diag = diagnose(model, scm, x, bg, actionable, n_samples=n_samples, rng=rng)
    a = list(actionable)
    pairs = list(combinations(a, 2))
    # instance-level interaction share (best-response and manuscript versions)
    br_abs = np.mean([abs(diag["br"][i, j]) for i, j in pairs])
    f1 = np.mean([diag["F1"][i] for i in a])
    stats = {
        "p_factual": float(model(x[None, :])[0]),
        "br_share": br_abs / (br_abs + f1 + 1e-12),
        "br_max": max(diag["br"][i, j] for i, j in pairs),
        "cg4_share": np.mean([abs(diag["cg4"][i, j]) for i, j in pairs])
        / (np.mean(np.abs(diag["cg1"][a])) + np.mean([abs(diag["cg4"][i, j]) for i, j in pairs]) + 1e-12),
        "diag_evals": diag["evals"],
    }
    rows = []
    for k in ks:
        o_ok, o_cost, o_T = oracle_targets(model, scm, x, a, k, bg, mad, bins=bins)
        for rule in rules:
            T = select_targets(rule, k, a, phi=phi, diag=diag, rng=rng)
            ok, cost, _, n_ev = recourse_on_targets(model, scm, x, T, bg, mad, bins=bins)
            rows.append({**stats, "k": k, "rule": rule, "targets": str(sorted(int(t) for t in T)),
                         "success": ok, "cost": cost, "search_evals": n_ev,
                         "oracle_success": o_ok, "oracle_cost": o_cost,
                         "oracle_targets": str(sorted(o_T)) if o_T else ""})
    return rows
