# Quantum Oracle Forensics — Team QuantMaxxing

**Event:** QC@IITI × IBM Qiskit Fall Fest 2026 (with QryptoNox)  
**Team:** QuantMaxxing · **Member:** Hussien Sulyman  
**Email:** hussiensulyman@gmail.com  
**Tag:** `v1.0-final-submit`

An adaptive Bayesian query strategy that identifies which of eight families a sealed, noisy quantum oracle belongs to and recovers its secret within a 60-query budget. The solver uses Fourier sampling, a Grover-iteration density test, and an exact Bayesian posterior over 10,795 Simon subspaces, with early stopping based on posterior confidence.

## Key Results (Validated on 1,200 tasks)

| Setting | Accuracy | Worst Family | Query Saving |
| :--- | :---: | :---: | :---: |
| Nominal noise (2% / 4%) | **98.8%** | 96% (grover) | 0.57 |
| Stress noise (4% / 8%) | **95.3%** | 87% (grover) | 0.50 |

## Repository Structure

- `solver.py` — Main solver: solve(n, m, budget, families, run)
- `oracle_sim.py` — Local noisy oracle simulator
- `evaluate.py` — Evaluation harness
- `baselines.py` — Baseline strategies (classical-only, single-algorithm, etc.)
- `test_spec.py` — Spec compliance checker (section 4 examples)
- `quick.py` — Quick test script (10 oracles/family)
- `run_all.py` — Batch evaluation runner
- `requirements.txt` — Python dependencies
- `oracle_report.pdf` — Technical report (3 pages)
- `oracle_ai_disclosure.pdf` — AI-use disclosure (1 page)
- `logs/` — Evaluation logs (results.json, summary.txt)

## Installation
```
pip install -r requirements.txt
```

**Dependencies:**
- qiskit>=1.0.0
- qiskit-aer>=0.14.0
- numpy>=1.24.0
- scipy>=1.10.0

## Running the Solver

The solver is invoked by the judges' harness as:

from solver import solve
family, secret = solve(n=8, m=1, budget=60, families=FAMILIES, run=run_fn)

For local testing, use the evaluation harness:

## Quick test (10 oracles/family)
```
python quick.py
````
## Full evaluation (150 oracles/family = 1200 tasks)
```
python evaluate.py 150
```
## Verify spec compliance (section 4 examples)
```
python test_spec.py
```
## Strategy Overview

- **m=1 branch:** Fourier sampling → Bayesian posterior → density test (balanced vs randfun) → Grover search with classical verification (constant vs grover)
- **m=8 branch:** Simon-style sampling → exact Bayesian posterior over 255 simon hypotheses, 10,795 simon2 subspaces, and perm → early stopping at 0.99 confidence

The solver uses slightly pessimistic noise rates (ε_model = 0.03, μ_model = 0.06) to remain robust under the doubled-noise stress setting.

## Reports

- **Technical report:** `oracle_report.pdf` (strategy, decision rules, thresholds, budget allocation, baselines, failure analysis)
- **AI-use disclosure:** `oracle_ai_disclosure.pdf`

## License

This work was developed during the QC@IITI × IBM Qiskit Fall Fest 2026 hackathon.
