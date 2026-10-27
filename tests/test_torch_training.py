import numpy as np
import pytest
import torch

from joint_data_BAM import make_targets, sample_batch
from joint_model_torch import JointBAMTorch
from joint_training_torch import joint_losses, train_step


def _batch(target):
    return {name: torch.as_tensor(np.asarray(value)[None], dtype=torch.float32)
            for name, value in target.items()}


def _predictions(d=3):
    return {"skeleton": torch.full((1, d, d), .5, requires_grad=True),
            "colliders": torch.full((1, d, d, d), .5, requires_grad=True),
            "direction": torch.full((1, d, d), .8, requires_grad=True),
            "order": torch.arange(d, dtype=torch.float32)[None].requires_grad_()}


def test_ambiguous_dag_labels_cannot_change_loss_or_direction_gradients():
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    target = _batch(make_targets(dag, dag_supervision=False))
    predictions = _predictions()
    first = joint_losses(target, predictions)
    target["dag"] = target["dag"].transpose(1, 2)
    second = joint_losses(target, predictions)
    torch.testing.assert_close(first["total"], second["total"])
    first["total"].backward()
    assert first["direction"].item() == first["order"].item() == 0
    assert predictions["direction"].grad.abs().sum().item() == 0
    assert predictions["order"].grad.abs().sum().item() == 0


def test_order_loss_does_not_force_incomparable_nodes_apart():
    dag = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    target = _batch(make_targets(dag, dag_supervision=True))
    p = _predictions()
    p["order"] = torch.tensor([[0., 0., 1.]])
    assert joint_losses(target, p)["order"].item() == 0
    p["order"] = torch.tensor([[1., 1., 0.]])
    assert joint_losses(target, p)["order"].item() == 2


def test_empty_graph_losses_are_finite_and_cpdag_only_needs_no_dag_heads():
    target = _batch(make_targets(np.zeros((3, 3), dtype=int), dag_supervision=True))
    p = _predictions()
    losses = joint_losses(target, p)
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(losses[name].item() == 0 for name in ("colliders", "direction", "order"))
    del p["direction"], p["order"]
    assert torch.isfinite(joint_losses(target, p, direction_weight=0, order_weight=0)["total"])


def test_loss_uses_only_unshielded_triples_and_counts_endpoints_once():
    dag = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    target = _batch(make_targets(dag))
    p = _predictions()
    p["colliders"] = torch.full((1, 3, 3, 3), .9)
    p["colliders"][0, 1, 2, 0] = .01  # Duplicate endpoint ordering is excluded.
    expected = -np.log(.9)
    assert joint_losses(target, p)["colliders"].item() == pytest.approx(expected)


def test_optimizer_updates_model_and_all_losses_are_finite():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = JointBAMTorch(n_channels=10, hidden_channels=8, collider_channels=4,
                         n_heads=2, data_layers=1, cov_layers=1)
    x, targets = sample_batch(2, 12, 4, mechanism="nonlinear_anm", seed=3)
    observations = torch.from_numpy(x)
    before = [p.detach().clone() for p in model.parameters()]
    losses = train_step(model, torch.optim.Adam(model.parameters(), lr=1e-4), observations, targets)
    assert all(torch.isfinite(value) for value in losses.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_smoke_cli_checks_checkpoint_and_decoders(tmp_path):
    from train_joint_torch import main
    result = main(["--smoke-test", "--output-dir", str(tmp_path / "smoke")])
    assert result["weights_changed"] and result["finite_losses_and_gradients"]
    assert result["checkpoint_max_error"] == 0
    assert result["cpdag_decoded"] and result["dag_preserves_cpdag"]
