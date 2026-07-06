# `trilinear` — registration-free lower-bound baseline

Resamples each stack onto the reconstruction grid by trilinear interpolation and
averages overlapping voxels. No learning, no super-resolution — the quality floor
every real method should beat.

```bash
conda run -n dev python methods/trilinear/run.py \
  --stacks data/sub01 --gt data/sub01/gt.nii.gz --out results/sub01/trilinear
```
Same input/output contract as the other methods. Use `--set normalize_stacks=per_stack`
for real multi-scanner data.
