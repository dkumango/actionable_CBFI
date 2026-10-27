"""
Symbolic check of the closed forms and signs in Section 3.3 and Appendix B.

Convention (as in the manuscript): g_S = h(x) - h(x^{do(S=v)}) is a REDUCTION, and the
interventional interaction is delta_ij = g_ij - g_i - g_j. Checks are on the logit scale.
Run: python check_formulas.py   (raises AssertionError if a manuscript formula is wrong)
"""
import sympy as sp

xi, xj, vi, vj, ui, uj, uk = sp.symbols("x_i x_j v_i v_j u_i u_j u_k")
wi, wj, wk, a, gam, beta, c = sp.symbols("w_i w_j w_k a gamma beta c")


def interaction(logit, do_i, do_j, do_ij):
    """delta_ij for reductions of `logit` (a function of a state dict)."""
    base = logit({})
    red = lambda d: base - logit(d)
    return sp.expand(red(do_ij) - red(do_i) - red(do_j))


# (b) path blocking: x_j = a x_i + u_j, additive logit w_i x_i + w_j x_j
def logit_b(do):
    Xi = do.get("i", xi)
    Xj = do.get("j", a * Xi + (xj - a * xi))          # abducted noise u_j = x_j - a x_i
    return wi * Xi + wj * Xj


d_b = interaction(logit_b, {"i": vi}, {"j": vj}, {"i": vi, "j": vj})
assert sp.simplify(d_b - wj * a * (vi - xi)) == 0, d_b
print("Lemma 1 (b):  delta =", sp.factor(d_b), " -> manuscript w_j a (v_i - x_i)  OK")


# (c) shared descendant: x_k = gamma x_i x_j + u_k, logit w_k x_k
xk = sp.Symbol("x_k")


def logit_c(do):
    Xi, Xj = do.get("i", xi), do.get("j", xj)
    return wk * (gam * Xi * Xj + (xk - gam * xi * xj))


d_c = interaction(logit_c, {"i": vi}, {"j": vj}, {"i": vi, "j": vj})
assert sp.simplify(d_c + wk * gam * (vi - xi) * (vj - xj)) == 0, d_c
print("Lemma 1 (c):  delta =", sp.factor(d_c), " -> manuscript -w_k gamma (v_i-x_i)(v_j-x_j)  OK")


# Proposition 1: adverse logit c - beta x_i x_j (other terms do not involve i, j)
def logit_p(do):
    Xi, Xj = do.get("i", xi), do.get("j", xj)
    return c - beta * Xi * Xj


d_p = interaction(logit_p, {"i": vi}, {"j": vj}, {"i": vi, "j": vj})
assert sp.simplify(d_p - beta * (vi - xi) * (vj - xj)) == 0, d_p
# expectation with v_i, v_j uncorrelated and mean zero: E[v_i v_j] = E[v_i] = E[v_j] = 0
e_p = sp.expand(d_p).subs(vi * vj, 0).subs({vi: 0, vj: 0})
assert sp.simplify(e_p - beta * xi * xj) == 0, e_p
print("Prop. 1 (i): C_G4 = beta E[(v_i-x_i)(v_j-x_j)] =", e_p, "  OK (sign +)")
# best joint reduction at x_i = x_j = 0
f2 = sp.expand((logit_p({}) - logit_p({"i": vi, "j": vj})).subs({xi: 0, xj: 0}))
assert sp.simplify(f2 - beta * vi * vj) == 0
print("Prop. 1 (ii): joint reduction at x=0 =", f2, "  OK")

# Appendix B: for linear h, delta_ij = +grad(h)^T Delta_{i via j}; here grad = (w_i, w_j) and
# Delta_{i via j} is the change of x_j caused by do(x_i): a (v_i - x_i) in coordinate j
via = sp.Matrix([0, a * (vi - xi)])
grad = sp.Matrix([wi, wj])
assert sp.simplify((grad.T * via)[0] - d_b) == 0
print("Appendix B:   delta = +grad(h)^T Delta_{i via j}  OK")
