"""Block-coordinate exact-LP initializer for the qq pooling model.

The product-quality map q(x) is bilinear in (lam, mu): fixing either block makes q
exactly affine (not an approximation) in the other. Alternating two elastic LPs -
one over the pool recipes (lam) with the blend (mu) fixed, one over the blend (mu)
with recipes (lam) fixed - converges toward a low- or zero-violation point far more
reliably than uniform random sampling, which has ~0 success probability once specs
include tight/near-equality bounds. IPOPT then polishes this point to full optimality.
"""
import numpy as np
from scipy.optimize import linprog


def _lam_block(p, mu):
    """Exact affine map q(lam) = q0 + Blin @ lam.ravel(), mu fixed."""
    M, I = p.M, p.I
    lam0 = np.zeros((M, I))
    x0 = np.r_[lam0.ravel(), mu.ravel()]
    _, _, q0, J = p.evaluate(x0)
    Blin = J[:, :M * I]
    return q0, Blin


def _mu_block(p, lam):
    """Exact affine map q(mu) = q0 + Blin @ mu.ravel(), lam fixed."""
    M, I = p.M, p.I
    mu0 = np.zeros_like(p.ml)
    x0 = np.r_[lam.ravel(), mu0.ravel()]
    _, _, q0, J = p.evaluate(x0)
    Blin = J[:, M * I:]
    return q0, Blin


def _solve_elastic_lam(p, mu, lam_hint=None):
    M, I = p.M, p.I
    lo = p.d['specs_lb'].ravel(); hi = p.d['specs_ub'].ravel()
    ns = lo.size
    q0, Blin = _lam_block(p, mu)
    nvar = M * I + 2 * ns
    c = np.zeros(nvar); c[M * I:] = 1.0
    Aeq = np.zeros((M, nvar))
    for m in range(M):
        Aeq[m, m * I:(m + 1) * I] = 1
    beq = np.ones(M)
    A_ub = np.zeros((2 * ns, nvar))
    A_ub[:ns, :M * I] = Blin; A_ub[:ns, M * I + ns:] = -np.eye(ns)
    A_ub[ns:, :M * I] = -Blin; A_ub[ns:, M * I:M * I + ns] = -np.eye(ns)
    b_ub = np.concatenate([hi - q0, q0 - lo])
    bounds = list(zip(p.ll.ravel(), p.lu.ravel())) + [(0, None)] * (2 * ns)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    if not res.success:
        return None, None
    lam = res.x[:M * I].reshape(M, I)
    viol = res.x[M * I:].sum()
    return lam, viol


def _solve_elastic_mu(p, lam):
    I, M, P = p.I, p.M, p.P
    lo = p.d['specs_lb'].ravel(); hi = p.d['specs_ub'].ravel()
    ns = lo.size
    q0, Blin = _mu_block(p, lam)
    nmu = P * (I + M)
    nvar = nmu + 2 * ns
    c = np.zeros(nvar); c[nmu:] = 1.0
    Aeq = np.zeros((P, nvar))
    for pp in range(P):
        Aeq[pp, pp * (I + M):(pp + 1) * (I + M)] = 1
    beq = np.ones(P)
    A_ub = np.zeros((2 * ns, nvar))
    A_ub[:ns, :nmu] = Blin; A_ub[:ns, nmu + ns:] = -np.eye(ns)
    A_ub[ns:, :nmu] = -Blin; A_ub[ns:, nmu:nmu + ns] = -np.eye(ns)
    b_ub = np.concatenate([hi - q0, q0 - lo])
    bounds = list(zip(p.ml.ravel(), p.mu.ravel())) + [(0, None)] * (2 * ns)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
    if not res.success:
        return None, None
    mu = res.x[:nmu].reshape(P, I + M)
    viol = res.x[nmu:].sum()
    return mu, viol


def alternating_init(p, rounds=8, seed=0):
    rng = np.random.default_rng(seed)
    x0 = p.initial(rng)
    lam = x0[:p.M * p.I].reshape(p.M, p.I)
    best_x, best_viol = None, np.inf
    for r in range(rounds):
        mu, vmu = _solve_elastic_mu(p, lam)
        if mu is None:
            break
        x = np.r_[lam.ravel(), mu.ravel()]
        _, true_viol = p.check(x)
        if true_viol < best_viol:
            best_x, best_viol = x.copy(), true_viol
        if true_viol < 1e-9:
            break
        lam2, vlam = _solve_elastic_lam(p, mu)
        if lam2 is None:
            break
        lam = lam2
        x = np.r_[lam.ravel(), mu.ravel()]
        _, true_viol = p.check(x)
        if true_viol < best_viol:
            best_x, best_viol = x.copy(), true_viol
        if true_viol < 1e-9:
            break
    return best_x, best_viol
