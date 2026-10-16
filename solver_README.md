# Quantum Oracle Forensics — solver
Install: `pip install -r requirements.txt` (Python 3.10+, Qiskit 2.x, numpy, scipy). Judges call `solve(n, m, budget, families, run)` in solver.py -> (family, secret).
Local tests (own simulator, oracle_sim.py): `python test_spec.py` (spec section-4 check values), `python quick.py`, `python evaluate.py 30` (nominal + stress), `python run_all.py ours|budget|base_<name>`.
Logs: logs/results.json and logs/summary.txt (accuracy, confusion matrices, mean queries, budget curve, baselines).
Strategy: see docs/oracle_report.pdf. Circuits use only h, x, mcx, the 'oracle' placeholder and measure. Nothing outside run() is read.
Note: the local simulator is ours; test against the official harness before submitting.
