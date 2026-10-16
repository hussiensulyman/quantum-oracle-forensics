"""Checks the spec's public examples against our oracle definitions (section 4)."""
import numpy as np
from oracle_sim import *

N=256
par=lambda v:bin(v).count('1')&1
lin=lambda x:par(178&x)^1

assert [lin(0),lin(1),lin(2),lin(255)]==[1,1,0,1]
s=105; sim=lambda x:min(x,x^s); assert [sim(5),sim(108),sim(105),sim(200)]==[5,5,0,161]
H=[0,15,49,62]; s2=lambda x:min(x^h for h in H); assert [s2(0),s2(15),s2(49),s2(62),s2(1),s2(200)]==[0,0,0,0,1,199]
pm=lambda x:(37*x+11)%256; assert [pm(0),pm(1),pm(2),pm(255)]==[11,48,85,230]

# every generated family has the stated property
rng=np.random.default_rng(1)
o=make_oracle('balanced',rng); assert o.table.sum()==128
o=make_oracle('simon',rng); s=o.secret; assert all(o.table[x]==o.table[x^s] for x in range(N)) and len(set(o.table))==N//2
o=make_oracle('simon2',rng); H=[0]+o.secret; assert all(o.table[x]==o.table[x^h] for x in range(N) for h in H) and len(set(o.table))==N//4
o=make_oracle('perm',rng); assert sorted(o.table)==list(range(N))

# noiseless circuit sanity: BV recovers a
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
o=Oracle([par(178&x)^1 for x in range(N)],8,1,'linear',[178,1]); run=make_run(o,0,0)
qc=QuantumCircuit(9,8); qc.x(8); qc.h(8); qc.h(range(8)); qc.append(Gate("oracle",9,[]),range(9)); qc.h(range(8)); qc.measure(range(8),range(8))

assert run(qc,5)=={format(178,'08b'):5}, run(qc,5)
print("spec checks passed")
