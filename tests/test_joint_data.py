"""Contracts for joint graph labels and the declared supervision regimes."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import joint_data_BAM as data
from graph_utils_BAM import validate_dag


def test_collider_targets_and_input_immutability():
    # 0 -> 1 <- 2, followed by 1 -> 3.
    dag = np.zeros((4, 4), dtype=np.float32)
    dag[0, 1] = dag[2, 1] = dag[1, 3] = 1
    original = dag.copy()
    targets = data.make_targets(dag, dag_supervision=True)
    np.testing.assert_array_equal(dag, original)
    assert targets["colliders"][0, 1, 2] == 1
    assert targets["colliders"][2, 1, 0] == 1
    assert targets["colliders"].sum() == 2
    assert targets["collider_mask"][0, 1, 3] == 1
    assert targets["colliders"][0, 1, 3] == 0
    assert targets["collider_mask"][0, 1, 0] == 0
    np.testing.assert_array_equal(targets["skeleton"], (dag + dag.T))
    assert targets["dag_mask"] == 1
    targets["dag"][0, 1] = 0
    np.testing.assert_array_equal(dag, original)


def test_shielded_colliders_are_not_training_triples():
    dag = np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]])
    targets = data.make_targets(dag)
    assert targets["collider_mask"].sum() == 0
    assert targets["colliders"].sum() == 0
    assert targets["dag_mask"] == 0


@pytest.mark.parametrize(
    "mechanism, expected_mask",
    [("linear_gaussian", 0), ("linear_nongaussian", 1), ("nonlinear_anm", 1)],
)
def test_regime_masks_shapes_and_reproducibility(mechanism, expected_mask):
    kwargs = dict(batch_size=2, num_samples=37, num_variables=6, mechanism=mechanism, seed=100)
    x, targets = data.sample_batch(**kwargs)
    repeat_x, repeat_targets = data.sample_batch(**kwargs)
    assert x.shape == (2, 37, 6)
    assert x.dtype == np.float32
    assert np.isfinite(x).all()
    np.testing.assert_allclose(x.mean(axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(x.std(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(x, repeat_x)
    np.testing.assert_array_equal(targets["dag_mask"], [expected_mask, expected_mask])
    for key, values in targets.items():
        assert values.dtype == np.float32
        np.testing.assert_array_equal(values, repeat_targets[key])
    assert targets["skeleton"].shape == (2, 6, 6)
    assert targets["colliders"].shape == targets["collider_mask"].shape == (2, 6, 6, 6)
    for adjacency in targets["dag"]:
        validate_dag(adjacency)


def test_observations_and_all_labels_share_variable_permutation(monkeypatch):
    dag = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)
    identifiable_columns = np.tile([10.0, 20.0, 30.0], (5, 1))
    monkeypatch.setattr(data, "_sample_adjacency", lambda *args: dag.copy())
    monkeypatch.setattr(data, "_sample_observations", lambda *args: identifiable_columns.copy())
    x, targets = data.sample_batch(1, 5, 3, seed=87, standardize=False)
    permutation = (x[0, 0] / 10 - 1).astype(int)
    permuted_dag = dag[np.ix_(permutation, permutation)]
    expected = data.make_targets(permuted_dag, dag_supervision=True)
    for key, value in expected.items():
        np.testing.assert_array_equal(targets[key][0], value)


def test_standardization_does_not_change_labels_or_rng_draws():
    kwargs = dict(batch_size=2, num_samples=25, num_variables=5, seed=19)
    raw, raw_targets = data.sample_batch(**kwargs, standardize=False)
    standardized, targets = data.sample_batch(**kwargs, standardize=True)
    expected = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    np.testing.assert_allclose(standardized, expected, atol=5e-7)
    for key in targets:
        np.testing.assert_array_equal(targets[key], raw_targets[key])


def test_mixed_regimes_and_explicit_override():
    kwargs = dict(
        batch_size=12,
        num_samples=15,
        num_variables=4,
        mechanism=("linear_gaussian", "nonlinear_anm"),
        seed=71,
    )
    _, targets = data.sample_batch(**kwargs)
    assert set(targets["dag_mask"]) == {0, 1}
    _, forced = data.sample_batch(**kwargs, dag_supervision=False)
    assert np.all(forced["dag_mask"] == 0)
    np.testing.assert_array_equal(targets["dag"], forced["dag"])


def test_empty_and_single_variable_graphs():
    for d in (1, 5):
        x, targets = data.sample_batch(2, 10, d, expected_degree=0, seed=21)
        assert np.isfinite(x).all()
        assert not np.any(targets["dag"])
        assert not np.any(targets["skeleton"])
        assert not np.any(targets["collider_mask"])
        assert not np.any(targets["colliders"])


def test_batch_stream_variable_shapes_and_seed():
    kwargs = dict(batch_size=2, num_samples=(5, 12), num_variables=(2, 6), steps_per_epoch=4, seed=51)
    stream = data.JointDataGenerator(**kwargs)
    repeated = data.JointDataGenerator(**kwargs)
    assert len(stream) == 4
    for i in range(4):
        x, targets = stream[i]
        other_x, other_targets = repeated[i]
        assert 5 <= x.shape[1] <= 12
        assert 2 <= x.shape[2] <= 6
        np.testing.assert_array_equal(x, other_x)
        np.testing.assert_array_equal(targets["dag"], other_targets["dag"])
    with pytest.raises(IndexError):
        stream[4]


def test_legacy_adapter_transposes_and_restores_global_rng(monkeypatch):
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)

    def legacy_sampler(child_parent, count):
        np.testing.assert_array_equal(child_parent, dag.T)
        return np.random.normal(size=(count, 3))

    monkeypatch.setitem(sys.modules, "chebyshev_numpy", SimpleNamespace(sample_from_dag_chebyshev=legacy_sampler))
    monkeypatch.setattr(data, "_sample_adjacency", lambda *args: dag.copy())
    before = np.random.get_state()
    x, targets = data.sample_batch(1, 8, 3, mechanism="chebyshev", seed=5)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    again, _ = data.sample_batch(1, 8, 3, mechanism="chebyshev", seed=5)
    np.testing.assert_array_equal(x, again)
    assert targets["dag_mask"][0] == 0


def test_rejects_invalid_dags_and_parameters():
    with pytest.raises(ValueError):
        data.make_targets([[0, 1], [1, 0]])
    for kwargs in (
        {"mechanism": "unknown"},
        {"expected_degree": -1},
        {"dag_supervision": 0.5},
        {"seed": 1, "rng": np.random.default_rng(1)},
    ):
        with pytest.raises(ValueError):
            data.sample_batch(1, 10, 3, **kwargs)
    with pytest.raises(ValueError):
        data.sample_batch(1, 1, 3)
