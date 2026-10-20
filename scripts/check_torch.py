"""Verify real CPU/MPS forward, backward, and optimizer execution."""

import argparse
import json
import os
import platform
import sys

import numpy as np
import torch


def check_device(device: str) -> dict[str, str | float]:
    """Check fixed float32 arithmetic against hand-derived values."""
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    weights = torch.tensor([[0.5], [1.0]], device=device, requires_grad=True)
    optimizer = torch.optim.SGD([weights], lr=0.001)
    predictions = inputs @ weights
    torch.testing.assert_close(predictions.detach().cpu(), torch.tensor([[2.5], [5.5]]))
    loss = predictions.square().mean()
    before = loss.item()
    loss.backward()
    if weights.grad is None:
        raise RuntimeError(f"Missing gradient on {device}")
    torch.testing.assert_close(weights.grad.cpu(), torch.tensor([[19.0], [27.0]]))
    optimizer.step()
    torch.testing.assert_close(
        weights.detach().cpu(), torch.tensor([[0.481], [0.973]]), rtol=1e-5, atol=1e-6
    )
    with torch.no_grad():
        after = (inputs @ weights).square().mean().item()
    if device == "mps":
        torch.mps.synchronize()
    if not after < before:
        raise RuntimeError(f"Loss did not decrease on {device}: {before} -> {after}")
    return {
        "status": "passed",
        "device": str(weights.device),
        "loss_before": before,
        "loss_after": after,
    }


def main() -> int:
    """Print the environment and results; optionally require a working MPS device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for a real MPS verification")
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    bridged = torch.from_numpy(array)
    torch.testing.assert_close(bridged, torch.tensor([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(bridged.numpy(), [1.0, 2.0, 3.0])
    mps_available = torch.backends.mps.is_available()
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "numpy_bridge": "passed",
        "dtype": "float32",
        "inputs": "fixed tensors; no random sampling",
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps_available,
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "cpu": check_device("cpu"),
        "mps": check_device("mps") if mps_available else {"status": "unavailable"},
    }
    print(json.dumps(report, indent=2))
    return int(args.require_mps and not mps_available)


if __name__ == "__main__":
    raise SystemExit(main())
