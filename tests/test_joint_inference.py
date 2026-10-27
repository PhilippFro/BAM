import numpy as np
import pytest

from graph_utils_BAM import collider_targets, dag_to_cpdag, validate_dag
from predict_joint_BAM import decode_predictions, predict


def _scores(dag):
    colliders, _ = collider_targets(dag)
    skeleton = np.maximum(dag, dag.T)
    return {"skeleton": skeleton * .98,
            "colliders": colliders * .98 + .01,
            "direction": np.where(dag, .95, .05),
            "order": np.arange(len(dag), dtype=float)}


def test_three_stage_outputs_are_consistent():
    dag = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    result = decode_predictions(_scores(dag), beam_width=32)
    np.testing.assert_array_equal(result["skeleton"], np.maximum(dag, dag.T))
    np.testing.assert_array_equal(result["cpdag"], dag_to_cpdag(dag))
    np.testing.assert_array_equal(dag_to_cpdag(result["dag"]), result["cpdag"])
    validate_dag(result["dag"])


def test_direction_predictions_do_not_change_cpdag():
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    scores = _scores(dag)
    first = decode_predictions(scores)
    scores["direction"] = scores["direction"].T
    scores["order"] *= -1
    second = decode_predictions(scores)
    np.testing.assert_array_equal(first["cpdag"], second["cpdag"])
    assert not np.array_equal(first["dag"], second["dag"])


def test_stopping_after_cpdag_does_not_expose_witness_as_dag():
    scores = _scores(np.zeros((3, 3), dtype=int))
    del scores["direction"], scores["order"]
    result = decode_predictions(scores, target="cpdag")
    assert "dag" not in result and "order" not in result
    with pytest.raises(ValueError, match="direction predictions"):
        decode_predictions(scores, target="both")


def test_predict_calls_network_once_for_all_stages():
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    class FakeNetwork:
        calls = 0
        def __call__(self, x, training=False):
            self.calls += 1
            assert x.shape == (1, 12, 3)
            return {key: value[None] for key, value in _scores(dag).items()}
    network = FakeNetwork()
    result, _ = predict(network, np.random.default_rng(0).normal(size=(12, 3)))
    assert network.calls == 1
    np.testing.assert_array_equal(dag_to_cpdag(result["dag"]), result["cpdag"])
