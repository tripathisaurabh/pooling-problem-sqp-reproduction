"""Trust-region sequential linear programming (SLP) solver for the paper's fixed-demand
qq pooling model, reusing Problem's exact evaluate()/check() machinery.

Only the product-quality constraint (lo <= q(x) <= hi) is nonlinear (bilinear in the pool
recipe and blend variables) and is linearized with an elastic L1 penalty each iteration.
The partition equality (Aeq@x=1) and pool-quality bounds (Apool@x in [lo,hi]) are exact
linear functions of x and are kept as hard constraints in every LP subproblem.
"""
import numpy as np
from scipy.optimize import linprog


def solve_slp(p, x0=None, seed=42, max_outer=200, trust0=0.3, trust_min=1e-10, trust_max=1.0,
              rho0=1e3, rho_max=1e8, tol=1e-9, verbose=False):
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0, _ = p.feasible_initial(rng)
        if x0 is None:
            x0 = p.initial(rng)
    x = x0.copy()
    lo = p.d['specs_lb'].ravel(); hi = p.d['specs_ub'].ravel()
    n = p.size
    ns = lo.size

    has_pool = p.active_pool.any()
    if has_pool:
        Apool = p.Apool[p.active_pool]
        poollo = p.poollo[p.active_pool]; poolhi = p.poolhi[p.active_pool]
        fin_lo = np.isfinite(poollo); fin_hi = np.isfinite(poolhi)

    def spec_violation(q):
        return float(np.sum(np.maximum(lo - q, 0)) + np.sum(np.maximum(q - hi, 0)))

    trust = trust0
    rho = rho0
    f, g, q, J = p.evaluate(x)
    viol = spec_violation(q)
    history = []
    for it in range(max_outer):
        lb_step = np.maximum(p.lower - x, -trust)
        ub_step = np.minimum(p.upper - x, trust)
        bad = lb_step > ub_step + 1e-12
        if np.any(bad):
            lb_step = np.where(bad, 0.0, lb_step)
            ub_step = np.where(bad, 0.0, ub_step)

        nvar = n + 2 * ns
        c = np.zeros(nvar)
        c[:n] = g
        c[n:n + ns] = rho
        c[n + ns:] = rho

        A_eq_rows = [np.hstack([p.Aeq, np.zeros((p.Aeq.shape[0], 2 * ns))])]
        b_eq = [np.zeros(p.Aeq.shape[0])]

        A_ub_rows = []
        b_ub = []
        # q + J@step <= hi + s_hi  ->  J@step - s_hi <= hi - q
        row = np.zeros((ns, nvar)); row[:, :n] = J; row[:, n + ns:] = -np.eye(ns)
        A_ub_rows.append(row); b_ub.append(hi - q)
        # q + J@step >= lo - s_lo  ->  -J@step - s_lo <= q - lo
        row = np.zeros((ns, nvar)); row[:, :n] = -J; row[:, n:n + ns] = -np.eye(ns)
        A_ub_rows.append(row); b_ub.append(q - lo)

        if has_pool:
            poolnow = Apool @ x
            if fin_hi.any():
                A = Apool[fin_hi]
                row = np.zeros((A.shape[0], nvar)); row[:, :n] = A
                A_ub_rows.append(row); b_ub.append(poolhi[fin_hi] - poolnow[fin_hi])
            if fin_lo.any():
                A = Apool[fin_lo]
                row = np.zeros((A.shape[0], nvar)); row[:, :n] = -A
                A_ub_rows.append(row); b_ub.append(poolnow[fin_lo] - poollo[fin_lo])

        A_eq = np.vstack(A_eq_rows); beq = np.concatenate(b_eq)
        A_ub = np.vstack(A_ub_rows); bub = np.concatenate(b_ub)
        bounds = list(zip(lb_step, ub_step)) + [(0, None)] * (2 * ns)

        res = linprog(c, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=beq, bounds=bounds, method='highs')
        if not res.success:
            trust *= 0.5
            if trust < trust_min:
                break
            continue

        step = res.x[:n]
        s_lo = res.x[n:n + ns]; s_hi = res.x[n + ns:]
        elastic_now = s_lo.sum() + s_hi.sum()
        predicted_reduction = -(g @ step) + rho * (viol - elastic_now)

        x_trial = x + step
        f_trial, g_trial, q_trial, J_trial = p.evaluate(x_trial)
        viol_trial = spec_violation(q_trial)
        merit_now = f + rho * viol
        merit_trial = f_trial + rho * viol_trial
        actual_reduction = merit_now - merit_trial

        step_norm = float(np.max(np.abs(step))) if n else 0.0
        ratio = actual_reduction / predicted_reduction if predicted_reduction > 1e-14 else -1.0

        accepted = ratio > 0.1 and predicted_reduction > -1e-12
        if verbose:
            print(f"it={it} trust={trust:.3g} rho={rho:.1e} f={f:.6f} viol={viol:.3e} "
                  f"pred={predicted_reduction:.3e} act={actual_reduction:.3e} ratio={ratio:.3f} "
                  f"accept={accepted}", flush=True)

        if accepted:
            x, f, g, q, J, viol = x_trial, f_trial, g_trial, q_trial, J_trial, viol_trial
            if ratio > 0.75 and step_norm > 0.9 * trust:
                trust = min(trust * 2, trust_max)
        else:
            trust *= 0.25

        if elastic_now > 1e-8 and viol > 1e-8:
            rho = min(rho * 2, rho_max)

        history.append({'it': it, 'objective': f, 'violation': viol, 'trust': trust, 'rho': rho,
                         'accepted': accepted})

        if trust < trust_min:
            break
        if predicted_reduction < tol and viol < 1e-9:
            break

    fval, true_viol = p.check(x)
    return {'x': x, 'objective': fval, 'violation': true_viol, 'iterations': len(history),
            'history': history}
