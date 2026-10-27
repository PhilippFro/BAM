"""Structural and differentiability tests for the shared neural model."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from joint_model_BAM import JointBAM


def make_model(**kwargs):
    tf.keras.utils.set_random_seed(1729)
    defaults = dict(
        n_channels=10,
        n_heads=2,
        hidden_channels=8,
        collider_channels=4,
        data_layers=1,
        cov_layers=1,
    )
    defaults.update(kwargs)
    return JointBAM(**defaults)


def observations(samples=17, variables=4):
    return np.random.default_rng(13).normal(size=(2, samples, variables)).astype("float32")


def test_shapes_symmetries_masks_and_initial_gate():
    model = make_model()
    result = {k: v.numpy() for k, v in model(observations()).items()}
    assert result["skeleton"].shape == (2, 4, 4)
    assert result["colliders"].shape == (2, 4, 4, 4)
    assert result["direction"].shape == (2, 4, 4)
    assert result["order"].shape == (2, 4)
    np.testing.assert_allclose(model.bam_gate.numpy(), 0.01)
    np.testing.assert_allclose(result["skeleton"], result["skeleton"].transpose(0, 2, 1))
    np.testing.assert_allclose(result["colliders"], result["colliders"].transpose(0, 3, 2, 1))
    np.testing.assert_allclose(
        result["direction"] + result["direction"].transpose(0, 2, 1),
        np.broadcast_to(1 - np.eye(4), (2, 4, 4)),
        atol=1e-6,
    )
    for key in ("skeleton", "direction"):
        np.testing.assert_array_equal(np.diagonal(result[key], axis1=1, axis2=2), 0)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                if len({a, b, c}) < 3:
                    np.testing.assert_array_equal(result["colliders"][:, a, b, c], 0)
    np.testing.assert_allclose(result["order"].mean(axis=-1), 0, atol=1e-6)
    # A directed head must actually be able to break exchange symmetry.
    assert np.max(abs(result["direction"] - result["direction"].transpose(0, 2, 1))) > 1e-4


def test_sample_invariance_and_variable_equivariance():
    model = make_model()
    x = observations()
    expected = {k: v.numpy() for k, v in model(x).items()}
    actual_samples = model(x[:, ::-1, :])
    for key in expected:
        np.testing.assert_allclose(actual_samples[key], expected[key], atol=3e-4, rtol=3e-4)
    perm = np.array([2, 0, 3, 1])
    actual_variables = model(x[:, :, perm])
    for key, value in expected.items():
        permuted = value
        for axis in range(1, value.ndim):
            permuted = np.take(permuted, perm, axis=axis)
        np.testing.assert_allclose(actual_variables[key], permuted, atol=3e-4, rtol=3e-4)


def test_both_branches_and_gate_receive_finite_nonzero_gradients():
    model = make_model()
    x = observations()
    with tf.GradientTape() as tape:
        result = model(x)
        # A nonsymmetric weighted objective exercises all readouts.
        loss = sum(
            tf.reduce_mean(value * tf.reshape(tf.linspace(0.1, 1.0, tf.size(value)), tf.shape(value)))
            for value in result.values()
        )
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    for variable, gradient in zip(variables, gradients):
        assert gradient is not None, variable.name
        assert bool(tf.reduce_all(tf.math.is_finite(gradient))), variable.name
    by_id = {id(variable): gradient for variable, gradient in zip(variables, gradients)}
    groups = [
        model.bam_attention[0].trainable_variables,
        model.directional_projection.trainable_variables,
        [model.bam_gate],
    ]
    for group in groups:
        assert sum(float(tf.reduce_sum(tf.abs(by_id[id(v)]))) for v in group) > 1e-10


def test_dynamic_shapes_in_one_tf_function_and_cpdag_only_mode():
    model = make_model(include_dag_heads=False)

    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float32)])
    def predict(x):
        return model(x)

    first = predict(observations(samples=11, variables=3))
    second = predict(observations(samples=19, variables=5))
    assert set(first) == {"skeleton", "colliders"}
    assert first["skeleton"].shape == (2, 3, 3)
    assert second["colliders"].shape == (2, 5, 5, 5)
    assert predict.experimental_get_tracing_count() <= 2


def test_weight_roundtrip_and_config(tmp_path):
    model = make_model()
    x = observations()
    expected = model(x)
    weights_path = str(tmp_path / "joint.weights.h5")
    model.save_weights(weights_path)
    restored = JointBAM.from_config(model.get_config())
    restored(x)
    restored.load_weights(weights_path)
    for key, value in restored(x).items():
        np.testing.assert_allclose(value, expected[key], atol=1e-6, rtol=1e-6)


def test_constant_observations_remain_finite():
    model = make_model()
    with tf.GradientTape() as tape:
        result = model(tf.ones([1, 8, 3]))
        loss = tf.reduce_sum(result["colliders"]) + tf.reduce_sum(result["skeleton"])
    assert all(bool(tf.reduce_all(tf.math.is_finite(value))) for value in result.values())
    gradients = tape.gradient(loss, model.trainable_variables)
    assert all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients if g is not None)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        JointBAM(n_channels=11, n_heads=2)
    with pytest.raises(ValueError, match="positive"):
        JointBAM(covariance_ridge=0)
