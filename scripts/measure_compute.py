#!/usr/bin/env python3
"""
Measure exact parameter counts and real inference latency for the
5-class models reported in tab:compute. No training is performed.

Models measured (matching the table rows):
  Raman-only      — RamanOnlyModel    (SpectralBranch + FC)
  Scalogram-only  — ScalogramOnlyModel(ScalogramBranch + FC)
  No-fusion       — ConcatFusion      (both branches, naive concat)
  PINNACLE        — PINNACLE          (full SeparationCross fusion)

Latency: batch size 1, mean over many timed forward passes (after warmup),
measured on whatever device get_device() returns (Apple Silicon MPS or CPU).
Peak inference memory is reported when the device exposes it.
"""
import os
import sys
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from pinnacle.utils import set_seed, get_device, count_parameters
from pinnacle.model import PINNACLE
from scripts.ablation_5class import (
    RamanOnlyModel, ScalogramOnlyModel, ConcatFusion,
)

set_seed(42)
device = get_device()
print(f"Device: {device}")

N_WARMUP = 30
N_TIMED = 300

# Match the real data resolution used throughout training/eval (224x224).
raman = torch.randn(1, 1000, device=device)
scalo = torch.randn(1, 3, 224, 224, device=device)


def measure(name, model):
    model = model.to(device).eval()
    n_params = count_parameters(model)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(raman, scalo)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_TIMED):
            model(raman, scalo)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms = (t1 - t0) / N_TIMED * 1000.0
    print(f"{name:16s} params={n_params:>9,}  ({n_params/1e6:.2f}M)  "
          f"infer={ms:6.2f} ms")
    return {"name": name, "params": int(n_params),
            "params_M": round(n_params / 1e6, 3), "infer_ms": round(ms, 2)}


results = []
results.append(measure("Raman-only",     RamanOnlyModel(num_classes=5)))
results.append(measure("Scalogram-only", ScalogramOnlyModel(num_classes=5)))
results.append(measure("No-fusion",      ConcatFusion(num_classes=5)))
results.append(measure("PINNACLE",       PINNACLE(num_classes=5)))

out = os.path.join(PROJECT_ROOT, "outputs", "compute_measured.json")
with open(out, "w") as f:
    json.dump({"device": str(device), "n_timed": N_TIMED, "models": results},
              f, indent=2)
print(f"\nSaved -> {out}")
