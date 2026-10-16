"""Baselines (spec section 9). Each takes (n,m,budget,families,run) like solve()."""
import numpy as np, solver
from solver import _Ctx,_Out,_classical,_samples

def classical_only(n,m,budget,families,run,seed=0):
    """Queries f at chosen inputs, never superposition. m=1: scan + re-verify ones; m=8: collision (birthday) voting."""
    rng=np.random.default_rng(seed); N=1<<n; ctx=_Ctx(run,budget)

    from qiskit import QuantumCircuit
    from solver import _gate

    def q8(x):
        qc=QuantumCircuit(n+8,8)
        for k in range(n):
            if (x>>k)&1: qc.x(k)
        qc.append(_gate(n,8),range(n+8)); qc.measure(range(n,n+8),range(8)); return _samples(ctx.go(qc,1))[0]
    pts=list(rng.permutation(N))

    if m==8:
        vals={}; 
        try:
            for x in pts[:budget]: vals[int(x)]=q8(int(x))

        except _Out: pass
        votes={}
        xs=list(vals)

        for i in range(len(xs)):
            for j in range(i+1,len(xs)):
                if vals[xs[i]]==vals[xs[j]] and vals[xs[i]]!=0: votes[xs[i]^xs[j]]=votes.get(xs[i]^xs[j],0)+1
        strong=sorted([d for d,c in votes.items() if c>=2],key=lambda d:-votes[d])

        if len(strong)>=3:
            for d1 in strong:
                for d2 in strong:
                    if d1<d2 and (d1^d2) in strong: return 'simon2',sorted([d1,d2,d1^d2])

        if strong: return 'simon',strong[0]
        if votes: 
            d=max(votes,key=votes.get)
            if votes[d]>=2: return 'simon',d
        return 'perm',None

    obs={}

    def rd(x): return int(next(iter(ctx.go(_classical(n,x),1)))[-1])
    pts=[0]+[1<<k for k in range(n)]+[int(x) for x in pts if x not in [0]+[1<<k for k in range(n)]]
    found=None

    try:
        for x in pts[:40]: obs[x]=rd(x)
        ones=[x for x in obs if obs[x]==1]
        if len(ones)<=4:                       # looks like constant-0 or grover: re-read the 1s
            for x in ones:
                if ctx.left()>=2 and rd(x)==1 and rd(x)==1 and found is None: found=x
        for x in pts[40:]:
            if ctx.left()<=0: break
            obs[x]=rd(x)

    except _Out: pass
    ys=np.array([obs[x] for x in obs]); xs=list(obs); frac=ys.mean() if len(ys) else 0
    best=(0,None)

    for a in range(1,N):
        for b in (0,1):
            agr=sum(((bin(a&x).count('1')&1)^b)==obs[x] for x in xs)
            if agr>best[0]: best=(agr,[a,b])

    if best[0]>=0.88*len(xs) and 0.2<frac<0.8 or best[0]>=0.93*len(xs) and best[1][0]: return 'linear',best[1]
    if frac>0.85: return 'constant',1
    if found is not None: return 'grover',found
    if frac<0.12: return 'constant',0

    return ('balanced',None) if frac>0.4 else ('randfun',None)

def single_algorithm(n,m,budget,families,run):
    """Always the Fourier-sampling circuit (BV / Simon), fixed 60 shots, posterior decision, no classical reads, no Grover."""
    ctx=_Ctx(run,budget)
    if m==8: return solver._solve_m8(n,m,ctx,{**solver.CFG,'adaptive':False,'m8_fixed':budget})
    cfg={**solver.CFG,'adaptive':False,'bv_fixed':budget}
    if n not in solver._BVM: solver._BVM[n]=solver._BVModel(n)
    obs=_samples(ctx.go(solver._bv(n),budget)); pr,ap=solver._BVM[n].post(obs); c=max(pr,key=pr.get)
    return {'bal':('balanced',None),'rf':('randfun',None),'lin':('linear',[int(np.argmax(ap))+1,0]),'z':('constant',0)}[c]

_RG=np.random.default_rng(2026)

def random_guess(n,m,budget,families,run):
    rng=_RG; f=families[int(rng.integers(len(families)))]
    N=1<<n; sec={'constant':int(rng.integers(2)),'linear':[int(rng.integers(1,N)),int(rng.integers(2))],'grover':int(rng.integers(N)),
      'simon':int(rng.integers(1,N)),'simon2':sorted(map(int,rng.integers(1,N,3))),}.get(f)
    return f,sec

def adaptivity_off(n,m,budget,families,run): return solver.solve(n,m,budget,families,run,cfg={'adaptive':False})
