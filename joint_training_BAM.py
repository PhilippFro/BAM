"""Masked objectives for joint skeleton, collider, and mechanism prediction.

All adjacency arrays use A[source, target]. DAG supervision is an explicit
per-dataset decision, never inferred from a network's prediction confidence.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf


def _binary_crossentropy(labels, probabilities):
    labels = tf.cast(labels, tf.float32)
    probabilities = tf.clip_by_value(tf.cast(probabilities, tf.float32), 1e-7, 1 - 1e-7)
    return -(labels * tf.math.log(probabilities)
             + (1 - labels) * tf.math.log1p(-probabilities))


def _masked_graph_mean(values, mask):
    """Equal weight per eligible graph, independent of d and triple count."""
    mask = tf.cast(mask, tf.float32)
    mask = tf.broadcast_to(mask, tf.shape(values))
    axes = tf.range(1, tf.rank(values))
    counts = tf.reduce_sum(mask, axis=axes)
    sums = tf.reduce_sum(values * mask, axis=axes)
    means = tf.math.divide_no_nan(sums, counts)
    eligible = tf.cast(counts > 0, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(means * eligible), tf.reduce_sum(eligible))


def joint_losses(targets, predictions, collider_weight=1.0,
                 direction_weight=1.0, order_weight=0.1, order_margin=1.0):
    """Compute the joint loss and its four separately reportable components.

    Skeleton BCE uses unordered pairs. Collider BCE is restricted to true
    unshielded triples and counts each endpoint pair once. Conditional direction
    BCE uses true skeleton edges; order hinge loss uses true directed edges.
    Both mechanism-dependent objectives require targets['dag_mask'] == 1.
    Graphs without eligible edges/triples contribute zero, without NaNs.
    """
    if min(collider_weight, direction_weight, order_weight) < 0 or order_margin <= 0:
        raise ValueError("Loss weights must be nonnegative and order_margin positive.")
    skeleton = tf.cast(targets["skeleton"], tf.float32)
    d = tf.shape(skeleton)[-1]
    index = tf.range(d)
    upper = tf.cast(index[:, None] < index[None, :], tf.float32)[None]
    loss_skeleton = _masked_graph_mean(
        _binary_crossentropy(skeleton, predictions["skeleton"]), upper)
    endpoint_upper = upper[:, :, None, :]
    collider_mask = tf.cast(targets["collider_mask"], tf.float32) * endpoint_upper
    loss_colliders = _masked_graph_mean(
        _binary_crossentropy(targets["colliders"], predictions["colliders"]),
        collider_mask)

    loss_direction = tf.constant(0.0)
    loss_order = tf.constant(0.0)
    if direction_weight > 0 or order_weight > 0:
        dag = tf.cast(targets["dag"], tf.float32)
        dag_mask = tf.cast(targets["dag_mask"], tf.float32)[:, None, None]
        tf.debugging.assert_greater_equal(dag_mask, 0.0)
        tf.debugging.assert_less_equal(dag_mask, 1.0)
        if direction_weight > 0:
            loss_direction = _masked_graph_mean(
                _binary_crossentropy(dag, predictions["direction"]),
                upper * skeleton * dag_mask)
        if order_weight > 0:
            ranks = tf.cast(predictions["order"], tf.float32)
            violations = tf.nn.relu(order_margin + ranks[:, :, None] - ranks[:, None, :])
            loss_order = _masked_graph_mean(violations, dag * dag_mask)
    total = (loss_skeleton + collider_weight * loss_colliders
             + direction_weight * loss_direction + order_weight * loss_order)
    return {"total": total, "skeleton": loss_skeleton, "colliders": loss_colliders,
            "direction": loss_direction, "order": loss_order}


def make_train_step(model, optimizer, **loss_options):
    """Create a traced update with finite-gradient checks and optimizer constraints."""
    @tf.function(reduce_retracing=True)
    def train_step(observations, targets):
        with tf.GradientTape() as tape:
            predictions = model(observations, training=True)
            losses = joint_losses(targets, predictions, **loss_options)
            tf.debugging.assert_all_finite(losses["total"], "Non-finite joint loss")
        gradients = tape.gradient(losses["total"], model.trainable_variables)
        pairs = [(grad, var) for grad, var in zip(gradients, model.trainable_variables)
                 if grad is not None]
        for grad, var in pairs:
            tf.debugging.assert_all_finite(grad, "Non-finite gradient: " + var.name)
        optimizer.apply_gradients(pairs)
        return losses
    return train_step
