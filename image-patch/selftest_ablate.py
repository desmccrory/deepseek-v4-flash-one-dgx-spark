#!/usr/bin/env python3
"""Selftest for the runtime refusal-direction ablation overlay.

Manual — not on the serve/boot path. From the repo root:

  ./start.sh selftest-ablate

No GPU, no compose.yml, no ./data required. Mounts files/direction_r1.pt
directly and reads IMAGE_DIGEST from start.sh.

Checks:
  - stock image model.py has no ablation hook
  - overlay with DSV4_ABLATE_FILE unset is inert (no buffer; forward gated)
  - overlay with FILE set: env parse, unit-norm load, layer gating
    (10-42 on, 0-9 and DSpark draft 43+ off), projection identity
    y.v == (1-lambda)*x.v (fp32-exact, bf16 within rounding),
    disabled-layer passthrough, missing FILE raises
"""
from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn as nn

OVERLAY = os.environ.get("DSV4_SELFTEST_OVERLAY", "/patch-test/model.py")
STOCK = os.environ.get(
    "DSV4_SELFTEST_STOCK",
    "/opt/vllm/vllm/models/deepseek_v4/nvidia/model.py",
)
DIRECTION = os.environ.get(
    "DSV4_SELFTEST_DIRECTION", "/models/files/direction_r1.pt"
)
KNOB_KEYS = ("DSV4_ABLATE_FILE", "DSV4_ABLATE_LAMBDA", "DSV4_ABLATE_LAYERS")


def load_py(path: str, name: str, env: dict[str, str]) -> object:
    for k in KNOB_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_on(mod, prefix: str):
    """Match DeepseekV4DecoderLayer.__init__ when DSV4_ABLATE_FILE is set."""
    d = nn.Module()
    d.hidden_size = 4096
    d._ablate_lambda = 0.0
    d.register_buffer(
        "_refusal_dir",
        torch.zeros(4096, dtype=torch.float32),
        persistent=False,
    )
    mod.DeepseekV4DecoderLayer._init_refusal_ablation(d, prefix)
    return d


# ---------------------------------------------------------------------------
# 1. Stock image file: no hook at all (what ABLATE=0 actually serves).
# spec_from_file_location makes inspect.getsource fail, so read the file.
# ---------------------------------------------------------------------------
stock = load_py(STOCK, "dsv4_stock", {})
assert not hasattr(stock, "_DSV4_ABLATE_FILE"), stock
assert not hasattr(stock.DeepseekV4DecoderLayer, "_ablate_refusal_direction")
assert not hasattr(stock.DeepseekV4DecoderLayer, "_init_refusal_ablation")
stock_text = open(STOCK, encoding="utf-8").read()
assert "_DSV4_ABLATE_FILE" not in stock_text
assert "_ablate_refusal_direction" not in stock_text
print("stock image model.py has no ablation hook")

# ---------------------------------------------------------------------------
# 2. Overlay with FILE unset: default-serve gate.
# ---------------------------------------------------------------------------
off = load_py(OVERLAY, "dsv4_overlay_off", {})
assert off._DSV4_ABLATE_FILE is None
# LAYERS still parse (default 10-42); FILE is the on/off gate.

overlay_text = open(OVERLAY, encoding="utf-8").read()
assert "self._refusal_dir = None" in overlay_text
# __init__ gate + two post-attn call sites in forward
assert overlay_text.count("if _DSV4_ABLATE_FILE is not None:") == 3

# Simulate the FILE-unset __init__ branch (full layer ctor needs vllm_config).
d_off_init = nn.Module()
d_off_init.hidden_size = 4096
d_off_init._ablate_lambda = 0.0
d_off_init._refusal_dir = None
assert d_off_init._refusal_dir is None
assert "_refusal_dir" not in d_off_init._buffers
# Forward does not call _ablate_refusal_direction when FILE is unset, so
# identity does not depend on the hook. Calling it with None must fail —
# that is why the forward gate exists.
x = torch.randn(7, 4096, dtype=torch.bfloat16)
try:
    off.DeepseekV4DecoderLayer._ablate_refusal_direction(d_off_init, x)
except TypeError:
    pass
else:
    raise AssertionError(
        "FILE-unset _ablate_refusal_direction must not be callable with None"
    )
print("FILE-unset gate OK  (_refusal_dir is None, forward gated, hook not required)")

# ---------------------------------------------------------------------------
# 3. Overlay with FILE set: env, load, gating, projection math.
# ---------------------------------------------------------------------------
on = load_py(
    OVERLAY,
    "dsv4_overlay_on",
    {
        "DSV4_ABLATE_FILE": DIRECTION,
        "DSV4_ABLATE_LAMBDA": "3.5",
        "DSV4_ABLATE_LAYERS": "10-42",
    },
)
assert on._DSV4_ABLATE_FILE == DIRECTION
assert on._DSV4_ABLATE_LAMBDA == 3.5
assert (on._DSV4_ABLATE_LAYER_LO, on._DSV4_ABLATE_LAYER_HI) == (10, 42)
print("env knobs OK")

L = on.DeepseekV4DecoderLayer
d15 = make_on(on, "model.layers.15")
v = d15._refusal_dir
assert isinstance(v, torch.Tensor) and v.shape == (4096,) and v.dtype == torch.float32
assert abs(v.norm().item() - 1.0) < 1e-5
assert d15._ablate_lambda == 3.5
assert "_refusal_dir" in d15._buffers
assert "_refusal_dir" not in d15.state_dict()
print("load+register OK  norm=", round(v.norm().item(), 6))

for pfx, want_on in [
    ("model.layers.5", False),
    ("model.layers.42", True),
    ("model.layers.43", False),
    ("model.layers.44", False),
]:
    d = make_on(on, pfx)
    is_on = d._ablate_lambda > 0 and d._refusal_dir.norm().item() > 0.5
    assert is_on is want_on, (pfx, is_on, want_on, d._ablate_lambda)
    assert isinstance(d._refusal_dir, torch.Tensor) and d._refusal_dir.shape == (4096,)
print("layer gating OK (5 off, 42 on, draft 43/44 off; all layers real tensors)")

y = L._ablate_refusal_direction(d15, x)
assert y.shape == x.shape and y.dtype == x.dtype
p_before = x.float() @ v
p_after = y.float() @ v
# y = x - 3.5 * (x.v) v  =>  y.v = (1 - 3.5) * x.v = -2.5 * x.v
err = (p_after + 2.5 * p_before).abs().max().item()
print(f"projection math OK  max|err|={err:.2e}")
assert err < 5e-2  # bf16 output rounding

x32 = torch.randn(7, 4096, dtype=torch.float32)
y32 = L._ablate_refusal_direction(d15, x32)
err32 = (y32 @ v + 2.5 * (x32 @ v)).abs().max().item()
print(f"fp32 identity OK  max err={err32:.2e}")
assert err32 < 1e-5

d_off = make_on(on, "model.layers.5")
y_off = L._ablate_refusal_direction(d_off, x)
assert torch.equal(y_off, x)
print("disabled-layer passthrough OK")

# Missing FILE must fail at init of an in-range layer (no silent stock boot).
missing = load_py(
    OVERLAY,
    "dsv4_overlay_missing",
    {
        "DSV4_ABLATE_FILE": "/no/such/direction.pt",
        "DSV4_ABLATE_LAMBDA": "3.5",
        "DSV4_ABLATE_LAYERS": "10-42",
    },
)
raised = False
try:
    make_on(missing, "model.layers.15")
except FileNotFoundError:
    raised = True
assert raised, "in-range layer must raise when DSV4_ABLATE_FILE is missing"
print("missing FILE errors OK")

print("ALL SELFTESTS PASSED")
