import io

import numpy as np
import pytest
import torch

from joint_model_torch import JointBAMTorch, symmetric_matrix_log


torch.set_num_threads(1)


def _model(**kwargs):
    torch.manual_seed(11)
    return JointBAMTorch(n_channels=10, data_layers=1, cov_layers=1,
                         hidden_channels=12, collider_channels=5, n_heads=2, **kwargs)


@pytest.mark.parametrize("offset", [0., 1e-10, 1e-6])
def test_log_gradient_at_repeated_or_nearby_eigenvalues(offset):
    matrix = torch.diag(torch.tensor([1., 1. + offset, 2.], dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(symmetric_matrix_log, (matrix,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_log_gradient_below_floor_and_off_diagonal_formula():
    matrix = torch.diag(torch.tensor([1e-5, 2e-5, .3], dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(symmetric_matrix_log, (matrix,), eps=1e-7, atol=2e-5, rtol=1e-4)
    identity = torch.eye(3, dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[1., 2., 3.], [2., 4., 5.], [3., 5., 6.]], dtype=torch.float64)
    gradient = torch.autograd.grad((symmetric_matrix_log(identity) * weights).sum(), identity)[0]
    torch.testing.assert_close(gradient, weights)


def test_shapes_symmetries_and_initial_gate():
    model = _model()
    x = torch.randn(2, 15, 4)
    result = model(x)
    assert {k: tuple(v.shape) for k, v in result.items()} == {
        "skeleton": (2, 4, 4), "colliders": (2, 4, 4, 4),
        "direction": (2, 4, 4), "order": (2, 4)}
    torch.testing.assert_close(model.bam_gate, torch.full((12,), .01))
    torch.testing.assert_close(result["skeleton"], result["skeleton"].transpose(1, 2))
    torch.testing.assert_close(result["colliders"], result["colliders"].transpose(1, 3))
    off = 1 - torch.eye(4)
    torch.testing.assert_close(result["direction"] + result["direction"].transpose(1, 2), off.expand(2, -1, -1))
    assert torch.count_nonzero(result["skeleton"].diagonal(dim1=-2, dim2=-1)) == 0
    for i in range(4):
        assert torch.count_nonzero(result["colliders"][:, i, i, :]) == 0
        assert torch.count_nonzero(result["colliders"][:, :, i, i]) == 0
        assert torch.count_nonzero(result["colliders"][:, i, :, i]) == 0


def test_sample_invariance_and_variable_equivariance():
    model = _model().double().eval()
    x = torch.randn(2, 12, 5, dtype=torch.float64)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    base = model(x)
    shuffled_samples = model(x[:, torch.randperm(12)])
    shuffled_variables = model(x[:, :, permutation])
    for key in base:
        torch.testing.assert_close(base[key], shuffled_samples[key], atol=2e-8, rtol=2e-7)
        expected = base[key][:, permutation]
        if key != "order":
            expected = expected[:, :, permutation]
        if key == "colliders":
            expected = expected[:, :, :, permutation]
        torch.testing.assert_close(expected, shuffled_variables[key], atol=2e-8, rtol=2e-7)


def test_all_branches_receive_finite_nonzero_gradients():
    model = _model()
    x = torch.randn(2, 12, 4, requires_grad=True)
    outputs = model(x)
    # Different weights prevent symmetric reductions from cancelling all
    # directional gradients. This tests branch connectivity, not loss design.
    objective = sum((value * torch.randn_like(value)).sum() for value in outputs.values())
    objective.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    groups = {"bam": [], "directional": [], "gate": []}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if name.startswith("bam_blocks"):
            groups["bam"].append(parameter.grad.abs().sum().item())
        if name.startswith(("source_features", "target_features")):
            groups["directional"].append(parameter.grad.abs().sum().item())
        if name == "bam_gate":
            groups["gate"].append(parameter.grad.abs().sum().item())
    assert all(sum(values) > 0 for values in groups.values())


@pytest.mark.parametrize("constant", [True, False])
def test_constant_and_duplicate_inputs_have_finite_backward(constant):
    model = _model()
    if constant:
        x = torch.zeros(2, 8, 4)
    else:
        column = torch.randn(2, 8, 1)
        x = column.expand(-1, -1, 4).clone()
    x.requires_grad_()
    outputs = model(x)
    objective = sum((value * torch.randn_like(value)).sum() for value in outputs.values())
    objective.backward()
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_dynamic_sizes_and_cpdag_only_mode():
    model = _model(include_dag_heads=False)
    for samples, variables in [(8, 3), (11, 5), (3, 1)]:
        result = model(torch.randn(2, samples, variables))
        assert set(result) == {"skeleton", "colliders"}
        assert result["skeleton"].shape == (2, variables, variables)
        assert all(torch.isfinite(value).all() for value in result.values())


def test_config_state_dict_roundtrip():
    model = _model().eval()
    observations = torch.randn(2, 10, 4)
    initial = model(observations)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = JointBAMTorch(**model.get_config())
    restored.load_state_dict(torch.load(stream, weights_only=True))
    restored.eval()
    for key, value in restored(observations).items():
        torch.testing.assert_close(value, initial[key], atol=0, rtol=0)


def test_bam_branch_preserves_positive_definiteness():
    model = _model().double()
    generator = torch.Generator().manual_seed(6)
    factors = torch.randn(2, 10, 5, 5, generator=generator, dtype=torch.float64)
    matrices = factors @ factors.transpose(-1, -2) + .1 * torch.eye(5, dtype=torch.float64)
    for block in model.bam_blocks:
        matrices = block(matrices)
        torch.testing.assert_close(matrices, matrices.transpose(-1, -2))
        assert torch.linalg.eigvalsh(matrices).min() > 0


def test_invalid_configuration_and_input():
    with pytest.raises(ValueError, match="divisible"):
        JointBAMTorch(n_channels=11, n_heads=2)
    with pytest.raises(ValueError, match="finite"):
        _model()(torch.full((1, 2, 3), float("nan")))
    with pytest.raises(ValueError, match="shape"):
        _model()(torch.randn(2, 3))
