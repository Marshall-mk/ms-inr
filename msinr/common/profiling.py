"""Compute/timing profiling: wall-clock, peak GPU memory, params, throughput,
and (best-effort) GPU energy via NVML.

Usage:
    prof = Profiler(device="cuda")
    with prof.section("reconstruct"):
        ... training ...
    with prof.section("inference"):
        ... sample the HR grid ...
    prof.add("num_parameters", count_parameters(model))
    profile_dict = prof.summary()
"""
from __future__ import annotations

import time
from contextlib import contextmanager


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


class _NVML:
    """Thin, optional wrapper over pynvml for energy/power (never fatal)."""

    def __init__(self, index: int = 0):
        self.ok = False
        self.h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.ok = True
        except Exception:
            self.pynvml = None

    def energy_mj(self):
        """Total energy consumed (millijoules) since driver load, or None."""
        if not self.ok:
            return None
        try:
            return int(self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h))
        except Exception:
            return None


class Profiler:
    def __init__(self, device: str = "cuda", nvml_index: int = 0):
        self.device = str(device)
        self.is_cuda = self.device.startswith("cuda")
        self.sections: dict = {}
        self.extra: dict = {}
        self._nvml = _NVML(nvml_index) if self.is_cuda else _NVML.__new__(_NVML)
        if not self.is_cuda:
            self._nvml.ok = False

    def _torch(self):
        import torch
        return torch

    @contextmanager
    def section(self, name: str):
        """Time a code block; also record peak GPU mem + energy delta for it."""
        torch = self._torch() if self.is_cuda else None
        if self.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        e0 = self._nvml.energy_mj()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.is_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rec = {"seconds": dt}
            if self.is_cuda:
                rec["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            e1 = self._nvml.energy_mj()
            if e0 is not None and e1 is not None:
                rec["energy_j"] = (e1 - e0) / 1000.0
                if dt > 0:
                    rec["avg_power_w"] = rec["energy_j"] / dt
            self.sections[name] = rec

    def add(self, key: str, value):
        self.extra[key] = value

    def throughput(self, name: str, n_items: int, key: str = "items_per_s"):
        """Record items/sec for a previously-timed section (e.g. voxels sampled)."""
        if name in self.sections and self.sections[name]["seconds"] > 0:
            self.extra[key] = n_items / self.sections[name]["seconds"]

    def summary(self) -> dict:
        return {"device": self.device, "sections": self.sections, **self.extra}
