import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from joint_data_BAM import make_targets, sample_batch
from joint_model_BAM import JointBAM
from joint_training_BAM import joint_losses, make_train_step


def _batch(target):
    return {k: np.asarray(v)[None] for k, v in target.items()}


def _predictions(d=3):
    return dict(skeleton=tf.fill((1, d, d), .5),
                colliders=tf.fill((1, d, d, d), .5),
                direction=tf.fill((1, d, d), .8),
                order=tf.constant([list(range(d))], dtype=tf.float32))


def test_nonidentifiable_graph_does_not_supervise_directions_or_order():
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    targets = _batch(make_targets(dag, dag_supervision=False))
    first = joint_losses(targets, _predictions())
    targets["dag"] = np.transpose(targets["dag"], (0, 2, 1))
    second = joint_losses(targets, _predictions())
    assert first["direction"].numpy() == first["order"].numpy() == 0
    np.testing.assert_allclose(first["total"], second["total"])


def test_order_loss_only_constrains_true_edges():
    dag = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    targets = _batch(make_targets(dag, dag_supervision=True))
    p = _predictions()
    p["order"] = tf.constant([[0., 0., 1.]])
    assert joint_losses(targets, p)["order"].numpy() == 0
    p["order"] = tf.constant([[1., 1., 0.]])
    assert joint_losses(targets, p)["order"].numpy() > 0


def test_empty_graph_has_finite_losses_and_zero_conditional_losses():
    targets = _batch(make_targets(np.zeros((3, 3), dtype=int), dag_supervision=True))
    losses = joint_losses(targets, _predictions())
    assert all(np.isfinite(v.numpy()) for v in losses.values())
    assert losses["colliders"].numpy() == losses["direction"].numpy() == losses["order"].numpy() == 0


def test_cpdag_only_predictions_do_not_require_direction_heads():
    targets = _batch(make_targets(np.zeros((3, 3), dtype=int)))
    p = _predictions()
    del p["direction"], p["order"]
    losses = joint_losses(targets, p, direction_weight=0, order_weight=0)
    assert np.isfinite(losses["total"])


def test_optimizer_update_is_finite_and_changes_weights():
    tf.keras.utils.set_random_seed(9)
    model = JointBAM(n_channels=10, hidden_channels=8, n_heads=2, data_layers=1, cov_layers=1)
    x, targets = sample_batch(2, 12, 4, mechanism="nonlinear_anm", seed=4)
    model(x)
    before = [v.numpy().copy() for v in model.trainable_variables]
    step = make_train_step(model, tf.keras.optimizers.Adam(1e-4, clipnorm=1))
    losses = step(x, targets)
    assert all(np.isfinite(v.numpy()) for v in losses.values())
    assert any(not np.array_equal(a, b.numpy()) for a, b in zip(before, model.trainable_variables))
