"""Loss equivalence checks plus opt-in real torchrun integration tests.

BAM_DISTRIBUTED_TESTS=1 python -m pytest -q tests/test_torch_distributed.py
requires loopback sockets (and two GPUs for the CUDA test).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from joint_data_BAM import make_targets
from joint_model_torch import JointBAMTorch
from joint_training_torch import _masked_graph_mean, joint_losses
from run_bam import runtime_environment


@pytest.mark.parametrize("mask", [
    [[0., 0., 0.], [0., 0., 0.], [1., 0., 0.], [1., 1., 1.]],
    [[1., 1., 0.], [1., 0., 0.], [0., 1., 1.], [0., 0., 0.]],
    [[0., 0., 0.]] * 4,
])
def test_global_mask_normalization_matches_combined_batch_gradients(mask):
    mask = torch.tensor(mask)
    combined = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    expected = _masked_graph_mean(combined, mask)
    expected.backward()
    counts = mask.sum(1)
    numerator = ((combined.detach() * mask).sum(1) / counts.clamp_min(1)).sum()
    stats = torch.stack((numerator, (counts > 0).float().sum()))
    local_gradients = []
    for part in (slice(0, 2), slice(2, 4)):
        local = combined.detach()[part].clone().requires_grad_()
        with patch("joint_training_torch.dist.all_reduce", side_effect=lambda t: t.copy_(stats)), \
             patch("joint_training_torch.dist.get_world_size", return_value=2):
            loss = _masked_graph_mean(local, mask[part], distributed=True)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        local_gradients.append(local.grad / 2)  # DDP averages rank gradients.
    torch.testing.assert_close(torch.cat(local_gradients), combined.grad)


def test_full_model_rank_gradients_match_combined_masked_objective():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = JointBAMTorch(n_channels=10, hidden_channels=8, collider_channels=4,
                         data_layers=1, cov_layers=1, n_heads=2)
    x = torch.randn(4, 12, 3)
    graphs = [np.zeros((3, 3), dtype=int), np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
              np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
              np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])]
    labels = [make_targets(g, dag_supervision=i >= 2) for i, g in enumerate(graphs)]
    targets = {key: torch.tensor(np.stack([t[key] for t in labels])) for key in labels[0]}
    reference = joint_losses(targets, model(x))
    reference["total"].backward()
    expected = [p.grad.clone() for p in model.parameters()]
    # Record detached sufficient statistics from both ranks, without a socket
    # transport. Actual torchrun coverage is below and enabled explicitly.
    stats = [[], []]
    for rank in range(2):
        part = slice(rank * 2, (rank + 1) * 2)
        with patch("joint_training_torch.dist.all_reduce", side_effect=lambda t: stats[rank].append(t.clone())), \
             patch("joint_training_torch.dist.get_world_size", return_value=2):
            joint_losses({k: v[part] for k, v in targets.items()}, model(x[part]), distributed=True)
    global_stats = [a + b for a, b in zip(*stats)]
    gradients = []
    for rank in range(2):
        model.zero_grad(set_to_none=True)
        part = slice(rank * 2, (rank + 1) * 2)
        values = iter(global_stats)
        with patch("joint_training_torch.dist.all_reduce", side_effect=lambda t: t.copy_(next(values))), \
             patch("joint_training_torch.dist.get_world_size", return_value=2):
            losses = joint_losses({k: v[part] for k, v in targets.items()}, model(x[part]), distributed=True)
        for key in losses:
            torch.testing.assert_close(losses[key], reference[key])
        losses["total"].backward()
        gradients.append([p.grad.clone() for p in model.parameters()])
    for first, second, wanted in zip(*gradients, expected):
        torch.testing.assert_close((first + second) / 2, wanted, atol=2e-6, rtol=2e-4)


def test_launcher_preserves_driver_paths_and_selects_venv_libraries():
    original = {"LD_LIBRARY_PATH": "/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/nvidia/lib64",
                "LD_PRELOAD": "/old/libtorch_python.so:/monitor/libmonitor.so", "PYTHONPATH": "/old/python",
                "PYTHONHOME": "/old", "PYTHON_EXEC": "/wrong/python", "CUDA_VISIBLE_DEVICES": "1,3"}
    env = runtime_environment(original, "/venv/torch/lib", "/venv/bin/python")
    assert env["LD_LIBRARY_PATH"] == "/venv/torch/lib:/usr/local/nvidia/lib64"
    assert env["LD_PRELOAD"] == "/monitor/libmonitor.so"
    assert env["PYTHON_EXEC"] == "/venv/bin/python"
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1" and env["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert "PYTHONHOME" in original


@pytest.mark.skipif(os.environ.get("BAM_DISTRIBUTED_TESTS") != "1",
                    reason="Set BAM_DISTRIBUTED_TESTS=1 on a host with local communication sockets")
@pytest.mark.parametrize("device,task", [("cpu", "both"), ("cpu", "cpdag"), ("cuda", "both")])
def test_real_two_worker_smoke_checkpoint_and_output_protection(tmp_path, device, task):
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA GPUs are required")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "distributed"
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=2",
               "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1",
               str(root / "train_joint_torch.py"), "--device", device, "--task", task,
               "--smoke-test", "--output-dir", str(output)]
    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo"
    completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "validation.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert report["world_size"] == 2 and report["global_batch_size"] == 4
    assert report["replica_max_error"] <= 1e-6 and report["checkpoint_max_error"] <= 1e-6
    assert report["cpdag_decoded"] and report["weights_changed"]
    if task == "both":
        assert report["dag_preserves_cpdag"]
    assert len(set(config["distributed"]["data_seeds"])) == 2
    before = (output / "model.pt").read_bytes()
    repeated = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
    assert repeated.returncode != 0
    assert "Output directory is not empty" in repeated.stdout + repeated.stderr
    assert (output / "model.pt").read_bytes() == before
