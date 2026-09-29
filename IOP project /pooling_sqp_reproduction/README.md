# Pooling problem reproduction: `qq` model and SQP style solver

## What the paper reports

Grothey, *On the effectiveness of sequential linear programming for the pooling problem*, compares filter-SLP, FilterSQP and IPOPT on animal-feed pooling data. Table 2 defines a “good” solution as feasible (maximum constraint violation below `1e-6`) and within **0.2%** of the best known objective in Table 1. The requested **0.02%** is ten times stricter and is tracked separately.

This implementation uses the paper's fixed-demand `qq` formulation (equations 5a–5h). SciPy's SLSQP is an SQP style implementation, but is **not FilterSQP** and will not reproduce the paper's solver timings or distribution of success over 500 starts. Pool recipes are shared across products, while each product can use permitted direct materials and pool outputs. Data `mu_*` rows are pools and `lam_*` rows are products, opposite the paper's Greek-letter variable names; the script maps them by row labels.

## Run

Requirements: Python 3.10+, NumPy, SciPy, and `python-docx` for the first `.docx` input.

```bash
python reproduce_pooling.py "inputs/af-1.txt (1).docx" --starts 5 --maxiter 150 --start-method random --seed 42 --output result.json
```

Use any complete `af-*.txt` input instead. `dt1.txt` is incomplete (missing `lam_lb` and `lam_ub`) and is included for provenance but cannot be solved as given. `af-3` and `af-5-fs` data were not among the supplied attachments. The script validates dimensions, bounds, fixed demand fractions, product and optional pool quality bounds, and cost. A failed SLSQP endpoint may be repaired by fixing its pool recipe and reoptimizing product blends with an LP; the repaired objective is only reported if fully feasible.

## Observed `af-1` result

| Item | Value |
|---|---:|
| Paper best-known objective (Table 1) | 296.44604 |
| Best feasible objective here (seed 42, one random start, LP repair) | 296.7587701766476 |
| Maximum constraint violation | 1.2526e-7 |
| Gap relative to paper best known | 0.1054931% |
| Paper's 0.2% criterion | Met |
| Requested 0.02% criterion | Not met |

The raw SLSQP endpoint for this run was infeasible (violation 0.00253). The fixed-pool LP repair made it feasible. Five further random starts with seed 7 yielded a best feasible cost of 297.19592 (0.25296% gap). These are small runs, not the paper's 500-start experiment. See the JSON records for iteration counts and failures.

The 0.02% target for `af-1` would require a feasible cost at most **296.505329208**. Meeting it may require many more starts, a stronger solver, or a closer reproduction of the authors' FilterSQP setup. An objective alone never certifies the requested result; feasibility must also hold.
