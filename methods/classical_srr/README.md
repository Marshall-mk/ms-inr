# `classical_srr` — classical LS-SRR baseline

Learning-free least-squares super-resolution reconstruction:
`min_x sum_k ||A_k x - y_k||^2 + lambda ||x||^2`, solved by conjugate gradient,
using the **same PSF forward operator** as the INR methods (materialized as a
sparse trilinear matrix). CPU/scipy — safe and deterministic.

## Run (isolated)
```bash
conda run -n dev python methods/classical_srr/run.py \
  --stacks data/sub01 --gt data/sub01/gt.nii.gz \
  --out results/sub01/classical_srr \
  --set reg_lambda=0.1 cg_maxiter=200 cg_tol=1e-5
```

Same input/output contract. Profile reports CG iterations and convergence.
