"""Quantum Oracle Forensics solver.  solve(n, m, budget, families, run) -> (family, secret)

Strategy (all decisions use query results only; see report):
 m=1 : Bernstein-Vazirani/Fourier sampling (1 query/shot, output in |->) -> Bayesian posterior over
       {constant-or-grover, linear, balanced, randfun}; then
       linear -> a from posterior, b from classical reads of f(0);
       constant/grover -> classical reads decide c=1; else sequential Grover search (k oracle calls/shot)
       with classical verification of each candidate; stop when posterior of 'grover' is small.
 m=8 : Simon/Fourier sampling of the input register (1 query/shot). Exact Bayesian posterior over
       255 simon hypotheses, 10795 simon2 subspaces and 'perm' (uniform), with a readout-noise kernel.
Sequential stopping once the joint posterior of the answer >= threshold, or budget is exhausted.
Noise rates are NOT given to the solver, so the likelihood uses slightly pessimistic fixed rates."""
import numpy as np
from scipy.special import logsumexp
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

EPS, MU = 0.03, 0.06          # model readout flip / oracle misfire (true nominal 0.02/0.04, stress 0.04/0.08)
CFG = dict(adaptive=True, bv_thr=0.99, bv_min=4, bv_batch=2, bv_max=16, grover_k=8, grover_stop=0.04,
           q_model=0.45, m8_thr=0.99, m8_batch=2, m8_fixed=30, bv_fixed=12, grover_fixed=4, g_j=3, g_thr=0.99, g_fixed=6)
LAST = {}

class _Out(Exception): pass
class _Ctx:
    def __init__(s, run, budget): s.run, s.budget, s.used = run, budget, 0
    def left(s): return s.budget - s.used
    def go(s, qc, shots=1):
        cost = qc.count_ops().get('oracle', 0) * shots
        if cost > s.left(): raise _Out()
        s.used += cost
        return s.run(qc, shots)
def _gate(n, m): return Gate("oracle", n + m, [])
def _samples(counts):
    out = []
    for k, v in counts.items(): out += [int(k.replace(' ', ''), 2)] * int(v)
    return out

# ---------------- circuits ----------------
def _bv(n):
    qc = QuantumCircuit(n + 1, n); qc.x(n); qc.h(n); qc.h(range(n))
    qc.append(_gate(n, 1), range(n + 1)); qc.h(range(n)); qc.measure(range(n), range(n)); return qc
def _simon(n, m):
    qc = QuantumCircuit(n + m, n); qc.h(range(n)); qc.append(_gate(n, m), range(n + m))
    qc.h(range(n)); qc.measure(range(n), range(n)); return qc
def _classical(n, x):
    qc = QuantumCircuit(n + 1, 1)
    for k in range(n):
        if (x >> k) & 1: qc.x(k)
    qc.append(_gate(n, 1), range(n + 1)); qc.measure(n, 0); return qc
def _grover(n, k):
    qc = QuantumCircuit(n + 1, n); qc.x(n); qc.h(n); qc.h(range(n))
    for _ in range(k):
        qc.append(_gate(n, 1), range(n + 1))
        qc.h(range(n)); qc.x(range(n)); qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    qc.measure(range(n), range(n)); return qc

def _gtest(n, j):
    """j Grover iterations, then H^n and measure: P(all-zero) = |<s|psi_j>|^2 (a density test, see report)."""
    qc = QuantumCircuit(n + 1, n); qc.x(n); qc.h(n); qc.h(range(n))
    for _ in range(j):
        qc.append(_gate(n, 1), range(n + 1))
        qc.h(range(n)); qc.x(range(n)); qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    qc.h(range(n)); qc.measure(range(n), range(n)); return qc
def _p0(theta, j, mu):
    import itertools
    sv = np.array([np.sin(theta), np.cos(theta)]); tot = 0.0
    for pat in itertools.product([0, 1], repeat=j):
        pr = np.prod([mu if b else 1 - mu for b in pat]); psi = sv.copy()
        for b in pat:
            if not b: psi = np.array([-psi[0], psi[1]])
            psi = 2 * sv * (sv @ psi) - psi
        tot += pr * (sv @ psi) ** 2
    return tot
class _GModel:
    """bal (exactly 128 ones -> theta=pi/4) vs randfun (density ~ N(0.3, 0.0286^2)) from j-iteration tests."""
    def __init__(s, n, j):
        N = 1 << n; K, _ = _kernel(n, EPS); nz = np.ones(N) / (N - 1); nz[0] = 0; Pnz = nz @ K
        mk = lambda p0: np.log(p0 * K[0] + (1 - p0) * Pnz)
        s.lb = mk(_p0(np.pi / 4, j, MU)); gr = np.linspace(-2, 2, 9); w = np.exp(-gr ** 2 / 2); w /= w.sum()
        s.lr = [mk(_p0(np.arcsin(np.sqrt(0.3 + 0.0286 * g)), j, MU)) for g in gr]; s.lw = np.log(w)
    def llr(s, obs):   # log P(obs|rf) - log P(obs|bal)
        o = np.asarray(obs)
        return logsumexp([lw + l[o].sum() for lw, l in zip(s.lw, s.lr)]) - s.lb[o].sum()

def _kernel(n, eps):
    N = 1 << n; pc = np.array([bin(i).count('1') for i in range(N)])
    d = pc[np.arange(N)[:, None] ^ np.arange(N)[None, :]]
    return eps ** d * (1 - eps) ** (n - d), pc

# ---------------- m = 1 ----------------
class _BVModel:
    def __init__(s, n):
        N = 1 << n; K, _ = _kernel(n, EPS); P0 = K[0]
        nz = np.ones(N) / (N - 1); nz[0] = 0; Pnz = nz @ K
        s.N = N; s.lz = np.log(P0); s.llin = np.log((1 - MU) * K[1:] + MU * P0[None, :])
        s.lbal = np.log((1 - MU) * Pnz + MU * P0)
        s.lrf = np.log((1 - MU) * (0.16 * P0 + 0.84 * Pnz) + MU * P0)
    def post(s, obs):
        o = np.asarray(obs); lla = s.llin[:, o].sum(axis=1)
        lw = np.array([np.log(.6) + s.lz[o].sum(), np.log(.2) + logsumexp(lla) - np.log(s.N - 1),
                       np.log(.2) + s.lbal[o].sum(), np.log(.2) + s.lrf[o].sum()])
        p = np.exp(lw - logsumexp(lw)); a = np.exp(lla - logsumexp(lla))
        return dict(z=p[0], lin=p[1], bal=p[2], rf=p[3]), a
_BVM = {}; _GM = {}
def _read(ctx, n, x):
    return int(next(iter(ctx.go(_classical(n, x), 1))).replace(' ', '')[-1])
_P1_HI = (1 - MU) * (1 - EPS) + MU * EPS   # P(read 1 | f(x)=1)
_P1_LO = EPS                                # P(read 1 | f(x)=0)

def _solve_m1(n, ctx, cfg):
    if n not in _BVM: _BVM[n] = _BVModel(n)
    M = _BVM[n]; obs = []; pr = None; a_post = None
    if cfg['adaptive']:
        while True:
            b = min(cfg['bv_batch'], ctx.left())
            if b <= 0 or len(obs) >= cfg['bv_max']: break
            obs += _samples(ctx.go(_bv(n), b)); pr, a_post = M.post(obs)
            if len(obs) >= cfg['bv_min']:
                if pr['z'] >= cfg['bv_thr'] or pr['lin'] * a_post.max() >= cfg['bv_thr']: break
                if pr['bal'] + pr['rf'] >= cfg['bv_thr']: break
    else:
        obs = _samples(ctx.go(_bv(n), min(cfg['bv_fixed'], ctx.left()))); pr, a_post = M.post(obs)
    if pr is not None and pr['bal'] + pr['rf'] >= 0.99 and pr['bal'] + pr['rf'] >= max(pr['z'], pr['lin']):
        J = cfg['g_j']; key = (n, J)
        if key not in _GM: _GM[key] = _GModel(n, J)
        G = _GM[key]; go = []; pb = pr['bal'] / (pr['bal'] + pr['rf']); lo0 = np.log((1 - pb) / pb); cnt = 0
        while ctx.left() >= J:
            if cfg['adaptive'] and len(go) >= 1 and abs(lo0 + G.llr(go)) >= np.log(cfg['g_thr'] / (1 - cfg['g_thr'])): break
            if not cfg['adaptive'] and len(go) >= cfg['g_fixed']: break
            go += _samples(ctx.go(_gtest(n, J), 1))
        if go: lo0 += G.llr(go)
        return ('randfun', None) if lo0 > 0 else ('balanced', None)
    if pr is None: return 'constant', 0
    cls = max(pr, key=pr.get)
    if cls == 'bal': return 'balanced', None
    if cls == 'rf': return 'randfun', None
    if cls == 'lin':
        a = int(np.argmax(a_post)) + 1; ones = zeros = 0
        lim = 6 if cfg['adaptive'] else 3
        while ones + zeros < lim and ctx.left() >= 1:
            r = _read(ctx, n, 0); ones += r; zeros += 1 - r
            if cfg['adaptive']:
                lr = (_P1_HI / _P1_LO) ** ones * ((1 - _P1_HI) / (1 - _P1_LO)) ** zeros
                if lr >= 99 or lr <= 1 / 99: break
        return 'linear', [a, int(ones > zeros)]
    # ---- z group: constant (c) or grover (t) ----
    pts = [0, (1 << n) - 1, 0xAA & ((1 << n) - 1), 0x55 & ((1 << n) - 1), 1, 2]
    ones = zeros = 0; i = 0; c1 = False
    lim = 2 if cfg['adaptive'] else 3
    while i < len(pts) and ctx.left() >= 1:
        r = _read(ctx, n, pts[i]); i += 1; ones += r; zeros += 1 - r
        lr = .5 * (_P1_HI / _P1_LO) ** ones * ((1 - _P1_HI) / (1 - _P1_LO)) ** zeros
        if cfg['adaptive'] and i >= 2 and (lr >= 99 or lr <= 1 / 99): break
        if not cfg['adaptive'] and i >= 3: break
    c1 = ones > zeros
    if c1: return 'constant', 1
    k = cfg['grover_k']; pg = 0.5; found = None; tries = 0
    while ctx.left() >= k + 1:
        if cfg['adaptive'] and pg < cfg['grover_stop']: break
        if not cfg['adaptive'] and tries >= cfg['grover_fixed']: break
        tries += 1
        cand = _samples(ctx.go(_grover(n, k), 1))[0]
        r = _read(ctx, n, cand)
        if r == 1 and ctx.left() >= 1:
            r2 = _read(ctx, n, cand)
            if r2 == 0 and ctx.left() >= 1: r2 = _read(ctx, n, cand)
            r = r2
        if r == 1 and found is None:
            found = cand
            if cfg['adaptive']: break
        q = cfg['q_model']; pg = pg * (1 - q) / (pg * (1 - q) + (1 - pg))
    if found is not None: return 'grover', int(found)
    return 'constant', 0

# ---------------- m = 8 ----------------
class _M8Model:
    def __init__(s, n):
        N = 1 << n; K, pc = _kernel(n, EPS); idx = np.arange(N)
        par = pc[idx[:, None] & idx[None, :]] & 1          # par[y, h] = y.h mod 2
        ind, sec, fam = [], [], []
        for sv in range(1, N): ind.append(par[:, sv] == 0); sec.append(sv); fam.append(0)
        for h1 in range(1, N):
            for h2 in range(h1 + 1, N):
                h3 = h1 ^ h2
                if h3 > h2: ind.append((par[:, h1] == 0) & (par[:, h2] == 0)); sec.append([h1, h2, h3]); fam.append(1)
        ind = np.array(ind, float); ind /= ind.sum(axis=1, keepdims=True)
        P = (1 - MU) * (ind @ K) + MU * K[0][None, :]
        Pp = (1 - MU) / N + MU * K[0]
        s.logP = np.log(np.vstack([P, Pp[None, :]])); s.sec = sec + [None]; s.fam = np.array(fam + [2])
        s.names = ['simon', 'simon2', 'perm']
_M8 = {}
def _solve_m8(n, m, ctx, cfg):
    if n not in _M8: _M8[n] = _M8Model(n)
    M = _M8[n]; LL = np.zeros(M.logP.shape[0]); nobs = 0
    sizes = np.array([(M.fam == f).sum() for f in range(3)])
    lprior = np.log(1 / 3) - np.log(sizes[M.fam])
    def posterior():
        lw = LL + lprior; return np.exp(lw - logsumexp(lw))
    while ctx.left() > 0:
        b = min(cfg['m8_batch'], ctx.left()) if cfg['adaptive'] else min(10, ctx.left(), cfg['m8_fixed'] - nobs)
        if b <= 0: break
        for o in _samples(ctx.go(_simon(n, m), b)): LL += M.logP[:, o]; nobs += 1
        if cfg['adaptive'] and nobs >= 3 and posterior().max() >= cfg['m8_thr']: break
    if nobs == 0: return 'perm', None
    i = int(np.argmax(posterior())); f = M.names[M.fam[i]]
    return f, M.sec[i]

def solve(n, m, budget, families, run, cfg=None):
    cfg = {**CFG, **(cfg or {})}; ctx = _Ctx(run, budget)
    try:
        res = _solve_m1(n, ctx, cfg) if m == 1 else _solve_m8(n, m, ctx, cfg)
    except _Out:
        res = ('constant', 0) if m == 1 else ('perm', None)
    LAST['used'] = ctx.used
    return res
