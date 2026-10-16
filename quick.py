import numpy as np, time, sys
from oracle_sim import *; import solver

rng=np.random.default_rng(5)

for fam in FAMILIES:
    ok=0;qs=[];t0=time.time()
    for i in range(10):
        o=make_oracle(fam,rng); run=make_run(o,0.04,0.02,rng)
        f,s=solver.solve(8,o.m,60,FAMILIES,run); ok+= (f==fam and s==o.secret); qs.append(run.state['q'])
    print(fam,ok,'/10 mean q',np.mean(qs),'t/oracle %.2f'%((time.time()-t0)/10))
