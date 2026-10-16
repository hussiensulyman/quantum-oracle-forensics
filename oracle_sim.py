"""Local oracle simulator for Quantum Oracle Forensics (section 3 of the spec).
make_run(oracle, p_mis, p_ro, rng, budget) -> run(circuit, shots) -> Qiskit-order counts.
Placeholder gate name 'oracle' acts on n input qubits then m output qubits: |x>|y> -> |x>|y xor f(x)>.
Each oracle application misfires (identity) w.p. p_mis; each measured bit flips w.p. p_ro. Query cost = k*shots."""
import numpy as np

class BudgetExceeded(Exception): pass

class Oracle:
    def __init__(self, table, n, m, family, secret):
        self.table=np.asarray(table,dtype=np.int64); self.n=n; self.m=m; self.family=family; self.secret=secret

# ---------- oracle families (section 3) ----------
def _par(v): return bin(int(v)).count('1') & 1

def make_oracle(family, rng, n=8):
    N=1<<n
    if family=='constant':
        c=int(rng.integers(2)); return Oracle(np.full(N,c),n,1,family,c)
    if family=='balanced':
        t=np.zeros(N,int); t[rng.permutation(N)[:N//2]]=1; return Oracle(t,n,1,family,None)
    if family=='linear':
        a=int(rng.integers(1,N)); b=int(rng.integers(2))
        return Oracle([_par(a&x)^b for x in range(N)],n,1,family,[a,b])
    if family=='grover':
        t=int(rng.integers(N)); tab=np.zeros(N,int); tab[t]=1; return Oracle(tab,n,1,family,t)
    if family=='randfun':
        return Oracle((rng.random(N)<0.3).astype(int),n,1,family,None)
    if family=='simon':
        s=int(rng.integers(1,N)); relabel=rng.permutation(N)
        return Oracle([relabel[min(x,x^s)] for x in range(N)],n,8,family,s)
    if family=='simon2':
        while True:
            h1,h2=int(rng.integers(1,N)),int(rng.integers(1,N))
            if h1!=h2: break
        H=sorted({h1,h2,h1^h2}); relabel=rng.permutation(N)
        full=[0]+H
        return Oracle([relabel[min(x^h for h in full)] for x in range(N)],n,8,family,H)
    if family=='perm':
        return Oracle(rng.permutation(N),n,8,family,None)
    raise ValueError(family)

FAMILIES=['constant','balanced','linear','grover','randfun','simon','simon2','perm']

# ---------- tiny statevector engine ----------
def _h(psi,q):
    v=psi.reshape(-1,2,1<<q); a=v[:,0,:].copy(); b=v[:,1,:].copy()
    v[:,0,:]=(a+b)/np.sqrt(2); v[:,1,:]=(a-b)/np.sqrt(2); return psi

def _x(psi,q):
    v=psi.reshape(-1,2,1<<q); v[:,[0,1],:]=v[:,[1,0],:]; return psi

def _z(psi,q):
    v=psi.reshape(-1,2,1<<q); v[:,1,:]*=-1; return psi

def _mcx(psi,qs,idx):
    ctrl,t=qs[:-1],qs[-1]; mask=np.ones(len(idx),bool)
    for c in ctrl: mask&=((idx>>c)&1)==1
    sel=idx[mask]; out=psi.copy(); out[sel]=psi[sel^(1<<t)]; return out

def make_run(oracle, p_mis=0.04, p_ro=0.02, rng=None, budget=60):
    rng=rng or np.random.default_rng(0); st={'q':0,'calls':0}; cache={}

    def run(circuit, shots):
        nq=circuit.num_qubits; nc=circuit.num_clbits; ops=[]; meas={}
        for ci in circuit.data:
            name=ci.operation.name; qs=[circuit.find_bit(q).index for q in ci.qubits]
            if name=='measure': meas[circuit.find_bit(ci.clbits[0]).index]=qs[0]
            elif name!='barrier': ops.append((name,qs))
        k=sum(1 for o in ops if o[0]=='oracle'); st['q']+=k*shots; st['calls']+=1

        if st['q']>budget: raise BudgetExceeded(st['q'])
        idx=np.arange(1<<nq)

        def oidx(qs):
            key=tuple(qs)
            if key not in cache:
                n,m=oracle.n,oracle.m; x=np.zeros(len(idx),np.int64)
                for j in range(n): x|=((idx>>qs[j])&1)<<j
                fx=oracle.table[x]; fm=np.zeros(len(idx),np.int64)
                for j in range(m): fm|=((fx>>j)&1)<<qs[n+j]
                cache[key]=idx^fm
            return cache[key]
        cl=np.zeros(len(idx),np.int64)

        for c,q in meas.items(): cl|=((idx>>q)&1)<<c
        mis=rng.random((shots,k))<p_mis if k else np.zeros((shots,0),bool)
        pats,inv=np.unique(mis,axis=0,return_inverse=True); inv=np.asarray(inv).reshape(-1)
        outs=np.zeros(shots,np.int64)

        for pi,pat in enumerate(pats):
            psi=np.zeros(1<<nq,complex); psi[0]=1; oi=0
            for name,qs in ops:
                if name=='h': psi=_h(psi,qs[0])
                elif name=='x': psi=_x(psi,qs[0])
                elif name=='z': psi=_z(psi,qs[0])
                elif name=='oracle':
                    if not pat[oi]: o=np.empty_like(psi); o[oidx(qs)]=psi; psi=o
                    oi+=1
                elif name in('cx','ccx') or name.startswith('mcx'): psi=_mcx(psi,qs,idx)
                else: raise NotImplementedError(name)
            pr=np.bincount(cl,weights=np.abs(psi)**2,minlength=1<<nc); pr/=pr.sum()
            sel=np.where(inv==pi)[0]; outs[sel]=rng.choice(1<<nc,size=len(sel),p=pr)
        
        flips=(rng.random((shots,nc))<p_ro)
        outs^=(flips*(1<<np.arange(nc))).sum(axis=1)
        counts={}
        
        for o in outs: s=format(int(o),f'0{nc}b'); counts[s]=counts.get(s,0)+1
        return counts
    run.state=st; return run
