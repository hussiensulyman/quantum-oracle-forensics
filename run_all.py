import json, os, sys, time, numpy as np, solver, baselines
from evaluate import *

F='logs/results.json'; res=json.load(open(F)) if os.path.exists(F) else {}
stage=sys.argv[1]; t0=time.time()

def save(): json.dump(res,open(F,'w'),indent=1,default=float)

def log(*a): print(*a,'t=%ds'%(time.time()-t0),flush=True)

if stage=='ours':
    for name,mis,ro in (('nominal',.04,.02),('stress',.08,.04)):
        r=suite(solver.solve,30,11,mis,ro); res['ours_'+name]=r; save(); log('ours',name,round(r['mean_acc'],3),round(r['worst'],2),round(r['saving'],3),{k:round(v,2) for k,v in r['acc'].items()})

if stage=='budget':
    res['budget_curve']={}
    for b in (20,30,40,50,60):
        r=suite(solver.solve,12,21,.04,.02,budget=b); res['budget_curve'][b]=dict(acc=r['mean_acc'],worst=r['worst'],per=r['acc']); save(); log('budget',b,round(r['mean_acc'],3))

if stage.startswith('base_'):
    nm=stage[5:]; fn=dict(classical_only=baselines.classical_only,single_algorithm=baselines.single_algorithm,adaptivity_off=baselines.adaptivity_off,random_guess=baselines.random_guess)[nm]
    r=suite(fn,10,31,.04,.02); res[stage]=r; save(); log(nm,round(r['mean_acc'],3),round(r['worst'],2),round(r['saving'],3),{k:round(v,2) for k,v in r['acc'].items()})
