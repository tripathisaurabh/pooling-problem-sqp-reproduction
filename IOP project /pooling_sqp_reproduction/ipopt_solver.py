"""IPOPT-based solve of the paper's fixed-demand qq pooling model, reusing Problem's
exact evaluate()/check(). IPOPT is one of the three solvers the paper itself compares
(filter-SLP, FilterSQP, IPOPT) and its interior-point method handles the many tight
(near-equality) product-quality specs in these instances far more robustly than SciPy's
active-set SLSQP, which stalls immediately on the larger af-* instances.
"""
import numpy as np
import cyipopt


class PoolingIpopt:
    def __init__(self, p):
        self.p = p
        lo = p.d['specs_lb'].ravel(); hi = p.d['specs_ub'].ravel()
        self.has_pool = p.active_pool.any()
        m_eq = p.Aeq.shape[0]
        self.m_eq = m_eq
        self.n_spec = lo.size
        if self.has_pool:
            self.Apool_active = p.Apool[p.active_pool]
            self.poollo = p.poollo[p.active_pool]
            self.poolhi = p.poolhi[p.active_pool]
        cl = [0.0] * m_eq + list(lo)
        cu = [0.0] * m_eq + list(hi)
        if self.has_pool:
            cl += list(self.poollo)
            cu += list(self.poolhi)
        self.cl = np.array(cl); self.cu = np.array(cu)
        self.m = len(cl)

    def objective(self, x):
        return self.p.evaluate(x)[0]

    def gradient(self, x):
        return self.p.evaluate(x)[1]

    def constraints(self, x):
        p = self.p
        _, _, q, _ = p.evaluate(x)
        parts = [p.Aeq @ x - p.ones, q]
        if self.has_pool:
            parts.append(self.Apool_active @ x)
        return np.concatenate(parts)

    def jacobian(self, x):
        p = self.p
        _, _, _, J = p.evaluate(x)
        parts = [p.Aeq, J]
        if self.has_pool:
            parts.append(self.Apool_active)
        return np.vstack(parts).ravel()


def solve_ipopt(p, x0=None, seed=42, max_iter=3000, tol=1e-9, print_level=0):
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0, _ = p.feasible_initial(rng)
        if x0 is None:
            x0 = p.initial(rng)

    obj = PoolingIpopt(p)
    nlp = cyipopt.Problem(n=p.size, m=obj.m, problem_obj=obj,
                           lb=p.lower, ub=p.upper, cl=obj.cl, cu=obj.cu)
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('max_iter', max_iter)
    nlp.add_option('tol', tol)
    nlp.add_option('print_level', print_level)
    nlp.add_option('constr_viol_tol', 1e-9)

    x, info = nlp.solve(x0)
    fval, viol = p.check(x)
    return {'x': x, 'objective': fval, 'violation': viol, 'status': info['status'],
            'status_msg': info['status_msg']}
