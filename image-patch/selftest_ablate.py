#!/usr/bin/env python3
r"""Selftest for the runtime refusal-direction ablation patch in
image-patch/vllm/models/deepseek_v4/nvidia/model.py.

Run inside the pinned image (no GPU needed):

  docker run --rm --entrypoint /opt/runtime-venv/bin/python \
    -e PYTHONPATH=/opt/vllm:/opt/sparkinfer \
    -v "$(pwd)/data:/models" \
    -v "$(pwd)/image-patch/vllm/models/deepseek_v4/nvidia/model.py:/patch-test/model.py:ro" \
    -v "$(pwd)/image-patch/selftest_ablate.py:/patch-test/test.py:ro" \
    "$(grep -oP '^\s*image:\s*\K\S+' compose.yml | head -1)" /patch-test/test.py

Checks: env-knob parsing, direction loading (unit-norm, non-persistent
buffer), layer gating (10-42 on, 0-9 and DSpark draft 43+ off), the
projection identity  y.v == (1 - lambda) * x.v  (fp32-exact, bf16 within
rounding), and disabled-layer passthrough.
"""
import os, importlib.util, sys
os.environ["DSV4_ABLATE_FILE"] = "/models/files/direction_r1.pt"
os.environ["DSV4_ABLATE_LAMBDA"] = "3.5"
os.environ["DSV4_ABLATE_LAYERS"] = "10-42"
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location(
    "dsv4_model_test", "/patch-test/model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert mod._DSV4_ABLATE_FILE == "/models/files/direction_r1.pt"
assert mod._DSV4_ABLATE_LAMBDA == 3.5
assert (mod._DSV4_ABLATE_LAYER_LO, mod._DSV4_ABLATE_LAYER_HI) == (10, 42)
print("env knobs OK")

L = mod.DeepseekV4DecoderLayer

def make(prefix):
    d = nn.Module()
    d.hidden_size = 4096
    d._ablate_lambda = 0.0
    d.register_buffer(
        "_refusal_dir",
        torch.zeros(4096, dtype=torch.float32),
        persistent=False,
    )
    L._init_refusal_ablation(d, prefix)
    return d

d15 = make("model.layers.15")
v = d15._refusal_dir
assert isinstance(v, torch.Tensor) and v.shape == (4096,) and v.dtype == torch.float32
assert abs(v.norm().item() - 1.0) < 1e-5
assert d15._ablate_lambda == 3.5
assert "_refusal_dir" in d15._buffers            # routed into _buffers, not __dict__
assert "_refusal_dir" not in d15.state_dict()  # non-persistent: excluded from state_dict
print("load+register OK  norm=", round(v.norm().item(), 6))

for pfx, want_on in [("model.layers.5", False), ("model.layers.42", True),
                     ("model.layers.43", False), ("model.layers.44", False)]:
    d = make(pfx)
    is_on = d._ablate_lambda > 0 and d._refusal_dir.norm().item() > 0.5
    assert is_on is want_on, (pfx, is_on, want_on, d._ablate_lambda, d._refusal_dir.norm().item())
    assert isinstance(d._refusal_dir, torch.Tensor) and d._refusal_dir.shape == (4096,)
print("layer gating OK (5 off, 42 on, draft 43/44 off; all layers real tensors)")

x = torch.randn(7, 4096, dtype=torch.bfloat16)
y = L._ablate_refusal_direction(d15, x)
assert y.shape == x.shape and y.dtype == x.dtype
p_before = (x.float() @ v)
p_after = (y.float() @ v)
# y = x - 3.5 * (x.v) v  =>  y.v = (1 - 3.5) * x.v = -2.5 * x.v
err = (p_after + 2.5 * p_before).abs().max().item()
print(f"projection math OK  max|err|={err:.2e}")
assert err < 5e-2  # bf16 output rounding

# fp32 path: the projection identity y.v == (1-lam) * x.v must be exact
x32 = torch.randn(7, 4096, dtype=torch.float32)
y32 = L._ablate_refusal_direction(d15, x32)
err32 = (y32 @ v + 2.5 * (x32 @ v)).abs().max().item()
print(f"fp32 identity OK  max err={err32:.2e}")
assert err32 < 1e-5

d_off = make("model.layers.5")
y_off = L._ablate_refusal_direction(d_off, x)
assert torch.equal(y_off, x)
print("disabled-layer passthrough OK")
print("ALL SELFTESTS PASSED")
