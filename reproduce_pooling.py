"""Reproduce the paper's fixed-demand qq pooling model with SciPy SLSQP.

Usage: python reproduce_pooling.py upload/af-1.dat --starts 5 --maxiter 500
The paper uses FilterSQP, not SciPy SLSQP; this reproduces its model and metrics,
not the authors' exact solver performance or random starts.
"""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, linprog

BEST = {'af-1':296.44604,'af-2':239334.02,'af-4':186097.11,'af-5':124003.48,
        'af-6':3074.23,'af-6-fs':2921.30,'af-7':151381.81,'af-7b':147285.00,
        'af-7b-fs':130290.06}


def read_data(path):
    path=Path(path)
    if path.suffix=='.docx':
        from docx import Document
        s='\n'.join(p.text for p in Document(path).paragraphs)
    else: s=path.read_text()
    sets = {k:v.split() for k,v in re.findall(r'set\s+(\w+)\s*:=\s*(.*?);',s,re.S)}
    dims = {k:len(v) for k,v in sets.items()}
    blocks = re.findall(r'param\s*:?\s*([^;]*?);',s,re.S)
    data = {}
    for block in blocks:
        head, body = block.split(':=',1)
        name = re.match(r'\s*(\w+)',head.lstrip(':')).group(1)
        if name == 'nuinrm':
            cols = head.split(':',1)[1].split()
            rows = [line.split() for line in body.splitlines() if line.strip()]
            assert len(rows)==dims['NU'] and all(len(r)==len(cols)+1 for r in rows)
            data[name] = np.array([[float(v) for v in r[1:]] for r in rows]).T
            assert cols==sets['RM']
        elif ':' in head:
            cols = head.split(':',1)[1].split()
            rows = [line.split() for line in body.splitlines() if line.strip()]
            assert all(len(r)==len(cols)+1 for r in rows), name
            data[name] = np.array([[float(v) for v in r[1:]] for r in rows])
            if name.startswith('specs_'): assert cols==sets['NU'] and [r[0] for r in rows]==sets['DE']
            elif name.startswith(('mu_','lam_')): assert cols==sets['RM']+sets['MI'] and [r[0] for r in rows]==(sets['MI'] if name.startswith('mu_') else sets['DE'])
            elif name.startswith('numi_'): assert cols==sets['NU'] and [r[0] for r in rows]==sets['MI']
        else:
            rows = [line.split() for line in body.splitlines() if line.strip()]
            data[name] = np.array([float(r[1]) for r in rows])
            assert [r[0] for r in rows]==sets['DE' if name=='tons' else 'RM']
    assert data['nuinrm'].shape==(dims['RM'],dims['NU'])
    assert data['mu_lb'].shape==(dims['MI'],dims['RM']+dims['MI'])
    assert data['lam_lb'].shape==(dims['DE'],dims['RM']+dims['MI'])
    return sets,data


class Problem:
    def __init__(self,path):
        self.sets,self.d=read_data(path)
        self.I=len(self.sets['RM']); self.M=len(self.sets['MI']); self.P=len(self.sets['DE']); self.N=len(self.sets['NU'])
        d=self.d; I=self.I; M=self.M; P=self.P; N=self.N
        # Data names are swapped relative to the paper: mu rows are pools; lam rows are products.
        # Pool columns include unused MI slots; preserve only raw-material columns.
        self.ll=d['mu_lb'][:,:I]; self.lu=d['mu_ub'][:,:I]
        self.ml=d['lam_lb']; self.mu=d['lam_ub']
        self.lower=np.r_[self.ll.ravel(),self.ml.ravel()]
        self.upper=np.r_[self.lu.ravel(),self.mu.ravel()]
        self.size=self.lower.size
        self.Aeq=np.zeros((M+P,self.size))
        for m in range(M): self.Aeq[m,m*I:(m+1)*I]=1
        for p in range(P): self.Aeq[M+p,M*I+p*(I+M):M*I+(p+1)*(I+M)]=1
        self.ones=np.ones(M+P)
        self.C=np.zeros((P*N,self.size)) # populated dynamically
        self.Apool=np.zeros((M*N,self.size))
        for m in range(M):
            self.Apool[m*N:(m+1)*N,m*I:(m+1)*I]=d['nuinrm'].T
        self.poollo=d.get('numi_lb',np.full((M,N),-np.inf)).ravel()
        self.poolhi=d.get('numi_ub',np.full((M,N),np.inf)).ravel()
        self.active_pool=np.isfinite(self.poollo)|np.isfinite(self.poolhi)
        assert np.all(self.lower<=self.upper)
        assert np.all(self.ll.sum(1)<=1+1e-9) and np.all(self.lu.sum(1)>=1-1e-9)
        assert np.all(self.ml.sum(1)<=1+1e-9) and np.all(self.mu.sum(1)>=1-1e-9)

    def split(self,x):
        a=self.M*self.I
        return x[:a].reshape(self.M,self.I),x[a:].reshape(self.P,self.I+self.M)

    def evaluate(self,x):
        lam,mu=self.split(x); R=self.d['nuinrm']; cost=self.d['RMcost']; tons=self.d['tons']
        mix=lam@R; mixcost=lam@cost
        q=mu[:,:self.I]@R+mu[:,self.I:]@mix
        c=mu[:,:self.I]@cost+mu[:,self.I:]@mixcost
        f=float(tons@c)
        gradlam=(mu[:,self.I:].T@tons[:,None])*cost[None,:]
        gradmu=tons[:,None]*np.r_[cost,mixcost][None,:]
        J=np.zeros_like(self.C)
        for p in range(self.P):
            # derivative by lam[m,i] = mu[p,I+m]*R[i,n]
            J[p*self.N:(p+1)*self.N,:self.M*self.I]=(mu[p,self.I:,None,None]*R[None,:,:]).transpose(2,0,1).reshape(self.N,self.M*self.I)
            J[p*self.N:(p+1)*self.N,self.M*self.I+p*(self.I+self.M):self.M*self.I+(p+1)*(self.I+self.M)]=np.r_[R,mix].T
        return f,np.r_[gradlam.ravel(),gradmu.ravel()],q.ravel(),J

    def initial(self,rng):
        def bounded_simplex(lo,hi):
            x=lo.copy(); room=1-x.sum()
            order=rng.permutation(len(x))
            for i in order:
                add=min(room,hi[i]-x[i]); x[i]+=add; room-=add
                if room<1e-12: break
            if room>1e-8: raise ValueError('Infeasible simplex bounds')
            return x
        return np.r_[np.array([bounded_simplex(l,u) for l,u in zip(self.ll,self.lu)]).ravel(),
                     np.array([bounded_simplex(l,u) for l,u in zip(self.ml,self.mu)]).ravel()]

    def check(self,x):
        f,_,q,_=self.evaluate(x)
        lo=self.d['specs_lb'].ravel(); hi=self.d['specs_ub'].ravel()
        pool=self.Apool@x
        violation=max(float(np.max(np.abs(self.Aeq@x-self.ones))),float(np.max(np.maximum(lo-q,0))),
                      float(np.max(np.maximum(q-hi,0))),float(np.max(np.maximum(self.lower-x,0))),
                      float(np.max(np.maximum(x-self.upper,0))))
        if self.active_pool.any():
            violation=max(violation,float(np.max(np.maximum(self.poollo[self.active_pool]-pool[self.active_pool],0))),
                          float(np.max(np.maximum(pool[self.active_pool]-self.poolhi[self.active_pool],0))))
        return f,violation

    def feasible_initial(self,rng,attempts=200,fixed_lam=None):
        """Find a feasible product blend by LP with the random pool recipes fixed."""
        I,M,P,N=self.I,self.M,self.P,self.N
        R=self.d['nuinrm']; c=self.d['RMcost']; tons=self.d['tons']
        for attempt in range(attempts):
            lam=(self.initial(rng)[:M*I].reshape(M,I) if fixed_lam is None else fixed_lam)
            mix=lam@R
            pool=mix.ravel()
            if self.active_pool.any() and (np.any(pool[self.active_pool]<self.poollo[self.active_pool]-1e-9) or np.any(pool[self.active_pool]>self.poolhi[self.active_pool]+1e-9)):
                continue
            qualities=np.r_[R,mix]
            price=np.r_[c,lam@c]
            nvars=P*(I+M)
            Aeq=np.zeros((P,nvars)); Aub=[]; bub=[]
            for p in range(P):
                sl=slice(p*(I+M),(p+1)*(I+M))
                Aeq[p,sl]=1
                for n in range(N):
                    row=np.zeros(nvars); row[sl]=qualities[:,n]
                    Aub.append(row); bub.append(self.d['specs_ub'][p,n])
                    Aub.append(-row); bub.append(-self.d['specs_lb'][p,n])
            lp=linprog(np.tile(price,P)*np.repeat(tons,I+M),A_ub=np.array(Aub),b_ub=np.array(bub),
                       A_eq=Aeq,b_eq=np.ones(P),bounds=list(zip(self.ml.ravel(),self.mu.ravel())),method='highs')
            if lp.success:
                return np.r_[lam.ravel(),lp.x],attempt+1
        return None,attempts

    def solve(self,starts=1,maxiter=300,seed=42,start_method="lp",target_gap=None,reference=None):
        rng=np.random.default_rng(seed); outcomes=[]; best=None; self.best_x=None
        for run in range(starts):
            x0,attempts=(self.feasible_initial(rng) if start_method=="lp" else (None,0))
            if x0 is None:
                print(f"No feasible fixed-pool LP in {attempts} attempts; using random start",flush=True)
                x0=self.initial(rng)
            else: print(f"Feasible LP start after {attempts} pool samples",flush=True)
            cache=[None,None]
            f0,v0=self.check(x0)
            if v0<1e-6:
                candidate={'start':run,'objective':f0,'max_violation':v0,'feasible':True,
                           'iterations':0,'seconds':0,'solver_success':True,'message':'Feasible fixed-pool LP start'}
                if best is None or f0<best['objective']: best=candidate; self.best_x=x0.copy()
            free=self.upper-self.lower>1e-12
            fixed=self.lower.copy()
            def expand(z):
                x=fixed.copy(); x[free]=z; return x
            def ev(z):
                if cache[0] is None or not np.array_equal(z,cache[0]): cache[:]=[z.copy(),self.evaluate(expand(z))]
                return cache[1]
            lo=self.d['specs_lb'].ravel(); hi=self.d['specs_ub'].ravel()
            cons=[{'type':'eq','fun':lambda x:self.Aeq@expand(x)-self.ones,'jac':lambda x:self.Aeq[:,free]},
                  {'type':'ineq','fun':lambda x:ev(x)[2]-lo,'jac':lambda x:ev(x)[3][:,free]},
                  {'type':'ineq','fun':lambda x:hi-ev(x)[2],'jac':lambda x:-ev(x)[3][:,free]}]
            if self.active_pool.any():
                A=self.Apool[self.active_pool]; low=self.poollo[self.active_pool]; high=self.poolhi[self.active_pool]
                finite=np.isfinite(low)
                if finite.any(): cons.append({'type':'ineq','fun':lambda x,A=A[finite],b=low[finite]:A@expand(x)-b,'jac':lambda x,A=A[finite]:A[:,free]})
                finite=np.isfinite(high)
                if finite.any(): cons.append({'type':'ineq','fun':lambda x,A=A[finite],b=high[finite]:b-A@expand(x),'jac':lambda x,A=A[finite]:-A[:,free]})
            t=time.monotonic()
            res=minimize(lambda x:ev(x)[0],x0[free],jac=lambda x:ev(x)[1][free],bounds=list(zip(self.lower[free],self.upper[free])),
                         constraints=cons,method='SLSQP',options={'maxiter':maxiter,'ftol':1e-9})
            f,v=self.check(expand(res.x))
            if v>=1e-6:
                repaired,_=self.feasible_initial(rng,attempts=1,fixed_lam=self.split(expand(res.x))[0])
                if repaired is not None:
                    fr,vr=self.check(repaired)
                    if vr<1e-6:
                        candidate={'start':run,'objective':fr,'max_violation':vr,'feasible':True,
                                   'iterations':int(res.nit),'seconds':round(time.monotonic()-t,3),
                                   'solver_success':False,'message':'Feasible LP repair at final pool recipe'}
                        if best is None or fr<best['objective']: best=candidate; self.best_x=repaired.copy()
            row={'start':run,'objective':f,'max_violation':v,'feasible':v<1e-6,
                 'iterations':int(res.nit),'seconds':round(time.monotonic()-t,3),'solver_success':bool(res.success),'message':res.message}
            outcomes.append(row); print(json.dumps(row),flush=True)
            if row['feasible'] and (best is None or f<best['objective']):
                best=row; self.best_x=expand(res.x).copy()
            if best is not None and reference is not None:
                gap=100*(best['objective']/reference-1)
                print(f'Best feasible gap after {run+1} start(s): {gap:.6f}%',flush=True)
                if target_gap is not None and gap<=target_gap:
                    print(f'Target {target_gap}% met; stopping early.',flush=True)
                    break
        return outcomes,best


