import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from graph_utils_BAM import dag_to_cpdag, validate_dag
from joint_model_torch import JointBAMTorch
from predict_joint_torch import load_model, predict


def small_model(**kwargs):
    torch.manual_seed(12)
    return JointBAMTorch(n_channels=8, n_heads=2, hidden_channels=8,
                         collider_channels=4, data_layers=1, cov_layers=1, **kwargs)


def test_one_forward_pass_all_stages_and_mode_restored():
    torch.set_num_threads(1)
    model = small_model().train()
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    observations = np.random.default_rng(4).normal(size=(16, 4)).astype(np.float32)
    original = observations.copy()
    graphs, scores = predict(model, observations, threshold=.25)
    handle.remove()
    assert calls == [1]
    assert model.training
    np.testing.assert_array_equal(observations, original)
    validate_dag(graphs["dag"])
    np.testing.assert_array_equal(dag_to_cpdag(graphs["dag"]), graphs["cpdag"])
    assert set(scores) == {"skeleton", "colliders", "direction", "order"}


def test_saved_checkpoint_loads_through_public_api(tmp_path):
    torch.set_num_threads(1)
    model = small_model()
    (tmp_path / "config.json").write_text(json.dumps({"model": model.get_config(), "convention": "source_target"}))
    torch.save(model.state_dict(), tmp_path / "model.pt")
    restored = load_model(tmp_path)
    observations = torch.randn(20, 5)
    graphs, scores = predict(model, observations)
    other_graphs, other_scores = predict(restored, observations)
    for key in scores:
        np.testing.assert_array_equal(scores[key], other_scores[key])
    for key in ("skeleton", "cpdag", "dag"):
        np.testing.assert_array_equal(graphs[key], other_graphs[key])


def test_cpdag_only_model_and_invalid_input():
    model = small_model(include_dag_heads=False)
    observations = np.random.default_rng(4).normal(size=(12, 4))
    result, _ = predict(model, observations, target="cpdag")
    assert "dag" not in result
    with pytest.raises(ValueError, match="direction predictions"):
        predict(model, observations)
    with pytest.raises(ValueError, match="finite"):
        predict(model, np.full((12, 4), np.nan))


def test_pytorch_path_does_not_import_tensorflow():
    subprocess.run([sys.executable, "-c",
                    "import sys; import predict_joint_torch; import generator_torch; "
                    "assert 'tensorflow' not in sys.modules"], check=True)
