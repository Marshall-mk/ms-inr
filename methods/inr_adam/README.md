# `inr_adam` — ablation baseline

Identical INR + PSF forward model as `inr_muon`, but trained with **Adam**. This
isolates the effect of Muon's high-rank updates (the paper's core claim) in the
multi-stack SRR setting. No extra dependencies beyond the `dev` env.

## Run (isolated)
```bash
conda run -n dev python methods/inr_adam/run.py \
  --stacks data/sub01 --gt data/sub01/gt.nii.gz \
  --out results/sub01/inr_adam --config configs/default.yaml \
  --set iters=3000 lr=0.001
```

Same input/output contract as `inr_muon`.
