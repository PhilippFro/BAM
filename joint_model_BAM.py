"""One BAM backbone with CPDAG and optional mechanism-dependent DAG heads.

All adjacency-like outputs use ``[source, target]``.  Collider tensors use
``[first_parent, child, second_parent]``.  Probabilities are predictions, not
causal-identifiability certificates; graph validity is handled by the decoders.

This module reuses the original TensorFlow/Keras 2 layers.  TensorFlow >= 2.16
therefore needs the matching ``tf-keras`` package, with legacy Keras selected
before importing TensorFlow.  Set that default here for standalone use.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

import custom_layers_BAM as cl


class JointBAM(tf.keras.Model):
    """Joint skeleton/collider/direction model with a gated BAM residual.

    Inputs have shape ``[batch, samples, variables]``.  The model is invariant
    to sample permutations and equivariant to variable permutations.  It has
    no learned variable identities or fixed-size positional embeddings.

    The directional path pools nonlinear contextual observation features and
    ordered source/target cross-moments.  BAM supplies a channel-gated residual
    *after* LogEig, projection, and normalization.  Nonlinear downstream heads
    let symmetric dependency information condition asymmetric predictions.

    Collider prediction explicitly materializes ``O(batch * d**3 * h)``
    activations, where ``h=collider_channels``.  Start with modest graph sizes;
    candidate-triple batching is a separate future scalability improvement.
    """

    def __init__(
        self,
        n_channels=20,
        data_layers=2,
        cov_layers=2,
        hidden_channels=32,
        collider_channels=16,
        n_heads=5,
        n_exp=3,
        gate_init=0.01,
        covariance_ridge=1e-4,
        include_dag_heads=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n_channels < 1 or n_heads < 1 or n_channels % n_heads:
            raise ValueError("n_channels must be positive and divisible by n_heads")
        if min(hidden_channels, collider_channels, n_exp) < 1:
            raise ValueError("hidden_channels, collider_channels and n_exp must be positive")
        if min(data_layers, cov_layers) < 0:
            raise ValueError("data_layers and cov_layers cannot be negative")
        if covariance_ridge <= 0:
            raise ValueError("covariance_ridge must be positive")
        self.n_channels = int(n_channels)
        self.data_layers = int(data_layers)
        self.cov_layers = int(cov_layers)
        self.hidden_channels = int(hidden_channels)
        self.collider_channels = int(collider_channels)
        self.n_heads = int(n_heads)
        self.n_exp = int(n_exp)
        self.gate_init = float(gate_init)
        self.covariance_ridge = float(covariance_ridge)
        self.include_dag_heads = bool(include_dag_heads)

        self.embedding = cl.layer_N_M_d_1_to_N_M_d_C_residual(self.n_channels)
        self.feature_attention = [
            cl.MultiHeadAttention_N_M_d_C_Feature(self.n_heads, self.n_channels)
            for _ in range(self.data_layers)
        ]
        self.sample_attention = [
            cl.MultiHeadAttention_N_M_d_C_Sample(self.n_heads, self.n_channels)
            for _ in range(self.data_layers)
        ]
        self.observation_ffn = [
            cl.layer_channels_dense_res_N_M_d_c(self.n_channels)
            for _ in range(self.data_layers)
        ]
        self.bam_attention = [
            cl.MultiHeadAttention_N_C_d_d_bilinear(self.n_heads)
            for _ in range(self.cov_layers)
        ]
        self.bam_activation = [
            cl.layer_N_C_d_d_spd_activation_scaled(N_exp=self.n_exp)
            for _ in range(self.cov_layers)
        ]

        dense = tf.keras.layers.Dense
        self.observation_norm = tf.keras.layers.LayerNormalization()
        self.source_features = dense(self.hidden_channels, activation="tanh")
        self.target_features = dense(self.hidden_channels, activation="tanh")
        self.node_features = dense(self.hidden_channels, activation="tanh")
        self.directional_projection = dense(self.hidden_channels, activation="gelu")
        self.directional_norm = tf.keras.layers.LayerNormalization()
        self.bam_projection = dense(self.hidden_channels, use_bias=False)
        self.bam_norm = tf.keras.layers.LayerNormalization()
        self.bam_gate = self.add_weight(
            name="bam_residual_gate",
            shape=(self.hidden_channels,),
            initializer=tf.keras.initializers.Constant(self.gate_init),
            trainable=True,
        )
        self.pair_hidden = dense(self.hidden_channels, activation="gelu")
        self.skeleton_readout = dense(1)

        # Each parent has the same role.  The child and endpoint-pair context
        # have separate projections, so the center role is not symmetrized away.
        self.collider_incoming = dense(self.collider_channels, use_bias=False)
        self.collider_endpoints = dense(self.collider_channels, use_bias=False)
        self.collider_center = dense(self.collider_channels)
        self.collider_readout = dense(1)

        if self.include_dag_heads:
            self.direction_readout = dense(1)
            self.order_hidden = dense(self.hidden_channels, activation="gelu")
            self.order_readout = dense(1)

    def _observation_encoding(self, inputs):
        h = self.embedding(inputs[..., None])
        for features, samples, ffn in zip(
            self.feature_attention, self.sample_attention, self.observation_ffn
        ):
            h = ffn(samples(features(h)))
        return h

    def _bam_features(self, h):
        cov = cl.data_N_M_d_c_to_cov_N_c_d_d(h)
        d = tf.shape(cov)[-1]
        cov = cov + self.covariance_ridge * tf.eye(d, dtype=cov.dtype)
        for attention, activation in zip(self.bam_attention, self.bam_activation):
            cov = activation(attention(cov))
        # The matrix logarithm is evaluated in float64 to reduce numerical
        # asymmetry/eigenvalue error, then returned to the network's dtype.
        matrix = tf.cast(0.5 * (cov + tf.linalg.matrix_transpose(cov)), tf.float64)
        values, vectors = tf.linalg.eigh(matrix)
        values = tf.maximum(values, tf.cast(self.covariance_ridge, values.dtype))
        log_matrix = tf.matmul(
            vectors * tf.math.log(values)[..., None, :], vectors, transpose_b=True
        )
        log_matrix = tf.cast(log_matrix, h.dtype)
        return tf.transpose(log_matrix, [0, 2, 3, 1])

    def _directional_features(self, h):
        normalized = self.observation_norm(h)
        source = self.source_features(normalized)
        target = self.target_features(normalized)
        # No [N,M,d,d,C] intermediate: reduce the sample axis inside einsum.
        cross = tf.einsum("nmic,nmjc->nijc", source, target)
        cross /= tf.cast(tf.shape(h)[1], h.dtype)
        nodes = tf.reduce_mean(self.node_features(normalized), axis=1)
        pair_shape = tf.shape(cross)
        first = tf.broadcast_to(nodes[:, :, None, :], pair_shape)
        second = tf.broadcast_to(nodes[:, None, :, :], pair_shape)
        pairs = self.directional_projection(tf.concat([first, second, cross], axis=-1))
        return self.directional_norm(pairs)

    @staticmethod
    def _node_context(pairs, off_diagonal):
        """Aggregate both endpoint roles without introducing variable IDs."""
        masked = pairs * off_diagonal[None, :, :, None]
        denominator = tf.maximum(tf.reduce_sum(off_diagonal, axis=-1), 1.0)
        outgoing = tf.reduce_sum(masked, axis=2) / denominator[None, :, None]
        incoming = tf.reduce_sum(masked, axis=1) / denominator[None, :, None]
        return tf.concat([outgoing, incoming], axis=-1)

    def call(self, inputs, training=None):
        del training  # Original BAM layers have no train/eval-specific behavior.
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        tf.debugging.assert_rank(inputs, 3)
        tf.debugging.assert_positive(tf.shape(inputs)[1], message="At least one sample is required")
        tf.debugging.assert_positive(tf.shape(inputs)[2], message="At least one variable is required")
        tf.debugging.assert_all_finite(inputs, "Observations must be finite")
        h = self._observation_encoding(inputs)
        directional = self._directional_features(h)
        bam = self.bam_norm(self.bam_projection(self._bam_features(h)))
        fused = directional + self.bam_gate * bam
        pairs = self.pair_hidden(fused)
        d = tf.shape(inputs)[2]
        off_diagonal = 1.0 - tf.eye(d, dtype=pairs.dtype)
        node_context = self._node_context(pairs, off_diagonal)

        skeleton_logits = tf.squeeze(self.skeleton_readout(pairs), axis=-1)
        skeleton_logits = 0.5 * (skeleton_logits + tf.linalg.matrix_transpose(skeleton_logits))
        skeleton = tf.sigmoid(skeleton_logits) * off_diagonal

        incoming = self.collider_incoming(pairs)
        endpoints = self.collider_endpoints(0.5 * (pairs + tf.transpose(pairs, [0, 2, 1, 3])))
        center = self.collider_center(node_context)
        # Axis order: [batch, parent_a, center, parent_c, channel].
        triple_features = (
            incoming[:, :, :, None, :]
            + tf.transpose(incoming, [0, 2, 1, 3])[:, None, :, :, :]
            + endpoints[:, :, None, :, :]
            + center[:, None, :, None, :]
        )
        collider_logits = tf.squeeze(self.collider_readout(tf.tanh(triple_features)), axis=-1)
        # Make endpoint symmetry explicit even under floating-point roundoff.
        collider_logits = 0.5 * (collider_logits + tf.transpose(collider_logits, [0, 3, 2, 1]))
        distinct = (
            off_diagonal[:, :, None]
            * off_diagonal[None, :, :]
            * off_diagonal[:, None, :]
        )
        outputs = {"skeleton": skeleton, "colliders": tf.sigmoid(collider_logits) * distinct}
        if self.include_dag_heads:
            logits = tf.squeeze(self.direction_readout(pairs), axis=-1)
            logits = logits - tf.linalg.matrix_transpose(logits)
            outputs["direction"] = tf.sigmoid(logits) * off_diagonal
            order = tf.squeeze(self.order_readout(self.order_hidden(node_context)), axis=-1)
            outputs["order"] = order - tf.reduce_mean(order, axis=-1, keepdims=True)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            n_channels=self.n_channels,
            data_layers=self.data_layers,
            cov_layers=self.cov_layers,
            hidden_channels=self.hidden_channels,
            collider_channels=self.collider_channels,
            n_heads=self.n_heads,
            n_exp=self.n_exp,
            gate_init=self.gate_init,
            covariance_ridge=self.covariance_ridge,
            include_dag_heads=self.include_dag_heads,
        )
        return config
