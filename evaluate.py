import numpy as np, json, sys, time
from oracle_sim import *; import solver, baselines

def trial(fn,fam,rng,mis,ro,budget=60,**kw):
    o=make_oracle(fam,rng); run=make_run(o,mis,ro,rng,budget)
    try: f,s=fn(8,o.m,budget,FAMILIES,run,**kw)

    except BudgetExceeded: return fam,None,False,budget+1
    ok=(f==fam and s==o.secret); return f,s,ok,run.state['q']

def suite(fn,per=30,seed=0,mis=0.04,ro=0.02,budget=60,**kw):
    rng=np.random.default_rng(seed); acc={};qs={};conf={}

    for fam in FAMILIES:
        r=[trial(fn,fam,rng,mis,ro,budget,**kw) for _ in range(per)]
        acc[fam]=np.mean([x[2] for x in r]); qs[fam]=[x[3] for x in r if x[2]]
        conf[fam]={}
        for x in r: conf[fam][x[0]]=conf[fam].get(x[0],0)+1

    allq=[q for f in FAMILIES for q in qs[f]]
    saving=np.mean([1-q/60 for q in allq]) if allq else 0

    return dict(acc=acc,mean_acc=float(np.mean(list(acc.values()))),worst=float(min(acc.values())),saving=float(saving),conf=conf,
                mean_q={f:float(np.mean(qs[f])) if qs[f] else None for f in FAMILIES})

if __name__=="__main__":
    per=int(sys.argv[1]) if len(sys.argv)>1 else 20
    for name,mis,ro in (('nominal',.04,.02),('stress',.08,.04)):
        r=suite(solver.solve,per,1,mis,ro); print(name,'acc %.3f worst %.3f saving %.3f'%(r['mean_acc'],r['worst'],r['saving']),{k:round(v,2) for k,v in r['acc'].items()})
