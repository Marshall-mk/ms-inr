"""Shared INR reconstruction driver used by the proposed method and the ablation.

``reconstruct_inr`` fits a coordinate MLP to the multi-stack observations through
the differentiable PSF forward model, then samples the trained field on the target
HR grid. The only difference between the proposed method and its ablation is the
``optimizer`` field ("muon" vs "adam"); everything else is identical, which makes
the head-to-head comparison clean.
"""
from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from .common.contracts import Volume, GridSpec, ReconResult
from .common.geometry import voxel_grid_world, CoordNormalizer
from .common.profiling import Profiler, count_parameters
from .forward.multistack import MultiStackForward
from .models.inr import build_inr
from .models.muon_setup import build_optimizer
from .data.dataset import (prepare_stack_tensors, MultiStackSampler,
                           recon_grid_from_stacks, normalize_stack_tensors)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_field_on_grid(model, normalizer: CoordNormalizer, grid: GridSpec,
                         device: str, chunk: int = 262144) -> np.ndarray:
    """Evaluate the INR on every voxel of ``grid`` -> (X,Y,Z) numpy array."""
    world = voxel_grid_world(grid.shape, grid.affine)          # (P,3)
    norm = normalizer.to_norm(world)
    norm_t = torch.as_tensor(norm, dtype=torch.float32, device=device)
    out = torch.empty(norm_t.shape[0], device=device)
    model.eval()
    with torch.no_grad():
        for i in range(0, norm_t.shape[0], chunk):
            out[i:i + chunk] = model(norm_t[i:i + chunk]).squeeze(-1)
    return out.reshape(grid.shape).cpu().numpy()


def _rank_snapshot(model) -> dict:
    if not hasattr(model, "get_detailed_matrix_info"):
        return {}
    infos = model.get_detailed_matrix_info()["layer_infos"]
    return {
        "stable_rank": [round(i["stable_rank"], 4) for i in infos],
        "effective_rank": [round(i["effective_rank"], 4) for i in infos],
        "spectral_norm": [round(i["spectral_norm"], 4) for i in infos],
    }


def _save_normalized_inputs(stacks, stack_tensors, roi, out_dir):
    """Dump each stack exactly as the model fits it: divided by its (mask-independent)
    norm_scale, plus a brain-masked copy if an ROI is used. If these have full T2
    contrast but the recon does not, the problem is the model, not the input."""
    import os
    from .common import io as _io
    from .common.roi import mask_at_world
    os.makedirs(out_dir, exist_ok=True)
    for st, t in zip(stacks, stack_tensors):
        norm = (st.data / t.norm_scale).astype(np.float32)
        _io.save_volume(Volume(norm, st.affine, st.name + "_norm"),
                        os.path.join(out_dir, f"{st.name}_norm.nii.gz"))
        if roi is not None:
            world = voxel_grid_world(st.shape, st.affine)
            inside = mask_at_world(roi, world).reshape(st.shape)
            _io.save_volume(Volume(norm * inside, st.affine, st.name + "_norm_masked"),
                            os.path.join(out_dir, f"{st.name}_norm_masked.nii.gz"))
    print(f"[debug] saved normalized inputs -> {out_dir}")


def reconstruct_inr(stacks, gt: Volume | None, cfg: dict, optimizer: str,
                    device: str | None = None) -> ReconResult:
    device = device or cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    set_seed(cfg.get("seed", 42))

    # target grid + shared coordinate frame
    grid = GridSpec.from_volume(gt) if gt is not None \
        else recon_grid_from_stacks(stacks, iso_mm=cfg.get("iso_mm", 1.0))

    # optional brain ROI: crop the grid to the brain bbox (concentrates the
    # normalizer/frequency budget on the brain) and drop non-brain samples.
    roi = None
    if cfg.get("roi_mask"):
        from .common import io as _io
        from .common.roi import crop_grid_to_mask
        roi = _io.load_volume(cfg["roi_mask"], name="roi")
        grid = crop_grid_to_mask(grid, roi, margin_mm=cfg.get("roi_margin_mm", 8.0))

    normalizer = CoordNormalizer.from_grid(grid.shape, grid.affine,
                                           pad_frac=cfg.get("pad_frac", 0.05))

    stack_tensors = prepare_stack_tensors(
        stacks, device=device, foreground_only=cfg.get("foreground_only", True),
        roi_mask=roi, normalize_q=cfg.get("normalize_q", 0.99))
    # normalize intensities to ~[0,1] so the phantom-tuned optimization transfers
    # to raw-valued MRI; rescale the reconstruction back afterwards.
    intensity_scale = normalize_stack_tensors(
        stack_tensors, mode=cfg.get("normalize_stacks", "global"),
        q=cfg.get("normalize_q", 0.99))
    # debug: dump the exact normalized (+ masked) inputs the model fits, so we can tell
    # whether a contrast issue comes from the input pipeline or the model.
    if cfg.get("debug_inputs_dir"):
        _save_normalized_inputs(stacks, stack_tensors, roi, cfg["debug_inputs_dir"])
    sampler = MultiStackSampler(stack_tensors, cfg.get("batch_per_stack", 4096))

    model = build_inr(
        cfg.get("model", "relu_ffn"), input_dim=3, output_dim=1,
        hidden_dim=cfg.get("hidden_dim", 256), num_layers=cfg.get("num_layers", 4),
        sigma=cfg.get("sigma", 6.0), num_freqs=cfg.get("num_freqs", 10),
        omega=cfg.get("omega", 30.0),
        mapping_size=cfg.get("mapping_size", 128)).to(device)

    # optional per-stack bias-field model (real low-field data): absorbs inter-stack
    # intensity disagreement so the field f doesn't fit it as noise.
    bias = None
    bias_mode = cfg.get("bias_field", "none")
    if bias_mode != "none":
        from .models.bias import PerStackBias
        bias = PerStackBias(len(stack_tensors), degree=cfg.get("bias_degree", 2),
                            mode=bias_mode).to(device)

    opt = build_optimizer(
        model, optimizer, lr=cfg.get("lr", 1e-3), muon_lr=cfg.get("muon_lr", 1e-2),
        weight_decay=cfg.get("weight_decay", 0.0),
        muon_weight_decay=cfg.get("muon_weight_decay", 0.0),
        extra_adam_params=(bias.parameters() if bias is not None else None))

    iters = cfg.get("iters", 3000)
    sched = None
    if cfg.get("scheduler", "cosine") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=1e-6)

    fwd = MultiStackForward(normalizer.center, normalizer.half_extent, device=device)

    # Speedups for the matmul-bound MLP (training only; inference/rank-tracking stay
    # on the fp32 base model). Batch shapes are constant across iters, so torch.compile
    # does not recompile. bf16 AMP needs no GradScaler.
    is_cuda = device.startswith("cuda")
    use_amp = bool(cfg.get("amp", True)) and is_cuda
    use_compile = bool(cfg.get("compile", True)) and is_cuda
    forward_model = model
    if use_compile:
        try:
            forward_model = torch.compile(model)
        except Exception as e:                       # pragma: no cover
            print(f"[recon] torch.compile disabled ({e})")
            forward_model, use_compile = model, False
    field = lambda pts: forward_model(pts).squeeze(-1)

    def amp_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    prof = Profiler(device)
    history = []
    with prof.section("reconstruct"):
        for it in range(iters):
            model.train()
            opt.zero_grad(set_to_none=True)
            batch = sampler.sample()
            with amp_ctx():
                preds = fwd.predict_batch(field, [(c, o, w) for c, o, w, _ in batch])
                loss = 0.0
                for i, (p, (coords, _, _, t)) in enumerate(zip(preds, batch)):
                    if bias is not None:            # y_k ~= exp(b_k(p)) * PSF(f)
                        p = bias.factor(i, fwd.to_norm(coords)) * p
                    loss = loss + F.mse_loss(p, t)
                loss = loss / len(stack_tensors)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if it % cfg.get("log_every", 100) == 0 or it == iters - 1:
                history.append({"iter": it, "loss": float(loss.detach()),
                                **_rank_snapshot(model)})

    with prof.section("inference"):
        recon = sample_field_on_grid(model, normalizer, grid, device,
                                     chunk=cfg.get("infer_chunk", 262144))
    recon = recon * intensity_scale     # back to input intensity units
    clamp_min = cfg.get("clamp_min", None)
    if clamp_min is not None:           # MRI is non-negative; removes negative speckles
        recon = np.clip(recon, float(clamp_min), None)
    if roi is not None:                 # output only the brain-masked region
        from .common.roi import mask_on_grid
        recon = recon * mask_on_grid(grid, roi)

    prof.add("num_parameters", count_parameters(model))
    prof.add("num_iters", iters)
    prof.add("intensity_scale", intensity_scale)
    prof.add("amp_bf16", use_amp)
    prof.add("torch_compile", use_compile)
    prof.add("bias_field", bias_mode)
    prof.throughput("inference", int(np.prod(grid.shape)), key="infer_voxels_per_s")
    prof.add("history", history)

    vol = Volume(data=recon, affine=grid.affine, name=f"recon_{optimizer}")
    return ReconResult(volume=vol, method=f"inr_{optimizer}", config=dict(cfg),
                       profile=prof.summary())
