import tensorflow as tf
tf.keras.backend.set_floatx('float32')
ddtype = tf.float32
lam_initial=1

class model_attention_final(tf.keras.Model):
    """
    Main class, full model with observational attention and bivariate attention in the SPD space

    Args:
        n_channels_main (int): Number of channels in the main layer.
        data_layers (int): Number of layers for attention between features and samples.
        cov_layers (int): Number of layers for bilinear attention and activation on the covariance matrix.
        inner_channels (int): Number of inner channels in the model.
        N_exp (int): Number of exponentiations for the SPD activation.
        N_heads (int): Number of heads for the multi-head attention mechanisms.
    """
    def __init__(self, n_channels_main=10, data_layers=2, cov_layers=4, inner_channels=10,N_exp=3, N_heads=5):
        super(model_attention_final, self).__init__()
        self.data_layers = data_layers
        self.cov_layers = cov_layers
        self.inner_channels = inner_channels
        self.n_channels_main = n_channels_main
        self.N_exp = N_exp
        self.N_heads=N_heads
        self.layer_N_M_d_1_to_N_M_d_C_residual = layer_N_M_d_1_to_N_M_d_C_residual(
            units_output=self.n_channels_main)
        l: int
        for l in range(1, self.data_layers + 1):
            setattr(self, f"layer_MH_attention_features_for_each_sample{l}",
                    MultiHeadAttention_N_M_d_C_Feature(num_heads=self.N_heads,inner_channels=self.n_channels_main))
            setattr(self, f"layer_attention_samples_for_each_feature{l}",
                    MultiHeadAttention_N_M_d_C_Sample(num_heads=self.N_heads, inner_channels=self.n_channels_main))
            setattr(self, f"layer_channels_dense_res_N_M_d_c{l}",
                    layer_channels_dense_res_N_M_d_c(inner_channels=self.n_channels_main))
        for l in range(1, self.cov_layers + 1):
            setattr(self, f"layer_N_C_d_d_bilinear_attention{l}",
                    MultiHeadAttention_N_C_d_d_bilinear(num_heads=5))
            setattr(self, f"layer_N_C_d_d_spd_activation{l}",
                    layer_N_C_d_d_spd_activation_scaled(N_exp=self.N_exp))
        self.layer_N_c_d_d_to_N_d_d_3_softmax = layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2()
    def call(self, inputs, **kwargs):
        M = tf.shape(inputs)[1]
        o1 = tf.expand_dims(inputs, 3)
        o2 = self.layer_N_M_d_1_to_N_M_d_C_residual(o1)
        out = o2
        for l in range(1, self.data_layers + 1):
            out = getattr(self, f"layer_MH_attention_features_for_each_sample{l}")(out)
            out = getattr(self, f"layer_attention_samples_for_each_feature{l}")(out)
            out = getattr(self, f"layer_channels_dense_res_N_M_d_c{l}")(out)

        cov1 = data_N_M_d_c_to_cov_N_c_d_d(out)

        out = cov1
        for l in range(1, self.cov_layers + 1):
            out = getattr(self, f"layer_N_C_d_d_bilinear_attention{l}")(out)
            out = getattr(self, f"layer_N_C_d_d_spd_activation{l}")(out)
        oout=[out,M]
        cov3 = self.layer_N_c_d_d_to_N_d_d_3_softmax(oout)
        return cov3

    def get_config(self):
        return {
            "data_layers": self.data_layers,
            "cov_layers": self.cov_layers,
            "inner_channels": self.inner_channels,
            "n_channels_main": self.n_channels_main,
            "N_exp": self.N_exp
        }

    def compute_output_shape(self, input_shape):
        return tf.concat([input_shape[0], input_shape[2], input_shape[2]], 0)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)


class layer_N_M_d_1_to_N_M_d_C_residual(tf.keras.layers.Layer):
    """
        Embedding of [M,d] matrices into [M,d,C] tensors.

        Args:
            units_output (int): Number of channels.
            activation (str): Activation function to be applied.
    """
    def __init__(self, units_output, activation="relu", **kwargs):
        self.activation_string = activation
        self.units_output = units_output
        self.activation = tf.keras.activations.get(activation)
        self.units = int(self.units_output)
        self.input_units = 1
        super().__init__(**kwargs)
        self.w = self.add_weight(
            shape=(self.input_units, self.units_output),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=ddtype,
            name=f"w"
        )
        self.w2 = self.add_weight(
            shape=(self.units_output, self.units_output),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=ddtype,
            name=f"w2"
        )
        self.llambda = self.add_weight(
            shape=(self.units_output,),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=ddtype,
            name=f"llambda"
        )

    def call(self, inputs, **kwargs):

        # Compute the output
        res = inputs + tf.multiply(
            tf.matmul(self.activation(tf.matmul(inputs, self.w)), self.w2), self.llambda)

        # Stack the outputs with M_matrix and d_matrix

        return res

    def get_config(self):
        return {
            "units_output": self.units_output,
            "activation": self.activation_string
        }

    def compute_output_shape(self, input_shape):
        return tf.concat([input_shape[0:2], tf.expand_dims(tf.cast(self.units_output, tf.int32), 0)], 0) + 2

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# noinspection PyAttributeOutsideInit
class layer_N_M_d_C_attention_features_for_each_sample(tf.keras.Model):
    """
        A layer that applies attention between attributes for each sample.

        Args:
            inner_channels (int): Number of inner channels.
            activation (str): Activation function to be applied.
    """
    def __init__(self, inner_channels, activation="relu", **kwargs):
        super(layer_N_M_d_C_attention_features_for_each_sample,self).__init__(**kwargs)
        self.activation_string = activation
        self.inner_channels = inner_channels
        self.activation = tf.keras.activations.get(activation)
        self.ln=tf.keras.layers.LayerNormalization()


    def build(self, input_shape):
        self.input_units = int(input_shape[3])
        self.w_keys = self.add_weight(
            shape=(self.input_units, self.inner_channels),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=self.dtype,
            name=f"w_keys"
        )
        self.w_queries = self.add_weight(
            shape=(self.input_units, self.inner_channels),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=self.dtype,
            name=f"w_queries"
        )
        self.w_values = self.add_weight(
            shape=(self.input_units, self.input_units),
            initializer=tf.keras.initializers.GlorotUniform(),
            #initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=self.dtype,
            name=f"w_values"
        )
        self.llambda = self.add_weight(
            shape=(self.input_units,),
            initializer=lam_init_eps(lam_initial),
            trainable=True,
            dtype=self.dtype,
            name=f"llambda"
        )

    # @tf.function(reduce_retracing=True)
    def call(self, inputs, **kwargs):

        inputs_normalized = self.ln(inputs)
        xwq = tf.matmul(inputs_normalized, self.w_queries)
        xwk = tf.matmul(inputs_normalized, self.w_keys)
        xwqkwx = tf.matmul(xwq, xwk, transpose_b=True)
        a = tf.nn.softmax(xwqkwx / tf.sqrt(tf.cast(self.inner_channels, tf.float32)))
        xwv = tf.matmul(inputs_normalized, self.w_values)
        axwv = tf.matmul(a, xwv)
        res = inputs + tf.multiply(axwv, self.llambda)
        return res

    def get_config(self):
        return {
            "inner_channels": self.inner_channels,
            "activation": self.activation_string
        }

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)


class layer_N_M_d_C_attention_samples_for_each_feature(tf.keras.Model):
    """
        A layer that applies attention between samples for each attribute.

        Args:
            inner_channels (int): Number of inner channels.
            activation (str): Activation function to be applied.
    """

    def __init__(self, inner_channels, activation="relu", **kwargs):
        super(layer_N_M_d_C_attention_samples_for_each_feature, self).__init__(**kwargs)
        self.activation_string = activation
        self.inner_channels = inner_channels
        self.activation = tf.keras.activations.get(activation)
        self.ln=tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        self.input_units = int(input_shape[3])
        self.w_keys = self.add_weight(
            shape=(self.input_units, self.inner_channels),
            initializer=tf.keras.initializers.GlorotUniform(),
            #initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=ddtype,
            name="w_keys"
        )
        self.w_queries = self.add_weight(
            shape=(self.input_units, self.inner_channels),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=ddtype,
            name="w_queries"
        )
        self.w_values = self.add_weight(
            shape=(self.input_units, self.input_units),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=ddtype,
            name="w_values"
        )
        self.llambda_ = self.add_weight(
            shape=(self.input_units,),
            initializer=lam_init_eps(lam_initial),
            trainable=True,
            dtype=ddtype,
            name="llambda"
        )


    # @tf.function(reduce_retracing=True)
    def call(self, inputs, **kwargs):
        inputs_normalized = self.ln(inputs)
        inputs_normalized = tf.transpose(inputs_normalized, perm=[0, 2, 1, 3])
        XWQ = tf.matmul(inputs_normalized, self.w_queries)
        XWK = tf.matmul(inputs_normalized, self.w_keys)
        XWQKWX = tf.matmul(XWQ, XWK, transpose_b=True)
        A = tf.nn.softmax(XWQKWX / tf.sqrt(tf.cast(self.inner_channels, tf.float32)))
        XWV = tf.matmul(inputs_normalized, self.w_values)
        AXWV = tf.matmul(A, XWV)
        AXWV = tf.transpose(AXWV, perm=[0, 2, 1, 3])

        res = inputs + tf.multiply(AXWV, self.llambda_)
        return res

    def get_config(self):
        return {"inner_channels": self.inner_channels, "activation": self.activation_string}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)


class layer_N_C_d_d_bilinear_attention_cov2cor_spd(tf.keras.layers.Layer):
    """
        A layer that first applies correlation normalization and then applies bilinear attention mechanism.
    """
    def __init__(self, **kwargs):
        super(layer_N_C_d_d_bilinear_attention_cov2cor_spd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_units = int(input_shape[1])
        self.w_keys = self.add_weight(
            shape=(self.input_units, self.input_units),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.RandomUniform(minval=0,maxval=2/ tf.cast(self.input_units, tf.float32)),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True,
            dtype=ddtype,
            name="w_keys"
        )
        self.w_queries = self.add_weight(
            shape=(self.input_units, self.input_units),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2 / tf.cast(self.input_units, tf.float32)),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True,
            dtype=ddtype,
            name="w_queries"
        )
        self.llambda = self.add_weight(
            shape=(self.input_units,),
            initializer=lam_init_eps(lam_initial),
            trainable=True,
            dtype=ddtype,
            name="llambda",
            constraint=tf.keras.constraints.NonNeg()
        )

    # @tf.function(reduce_retracing=True)
    def call(self, inputs, **kwargs):
        dia = tf.abs(tf.linalg.diag_part(inputs))
        dia = tf.maximum(dia, 0.0001)
        di = tf.sqrt(dia)
        d_inv = tf.linalg.diag(1 / di)
        cor = tf.matmul(tf.matmul(d_inv, inputs), d_inv)

        cor = tf.minimum(cor, 1)
        cor = tf.maximum(cor, -1)
        X=cor

        inputs_cov2cor = tf.transpose(X, perm=[0, 2, 3, 1])
        XWQ = tf.matmul(inputs_cov2cor, self.w_queries)
        XWK = tf.matmul(inputs_cov2cor, self.w_keys)
        XWQt = tf.transpose(XWQ, perm=[0, 3, 1, 2])
        XWKt = tf.transpose(XWK, perm=[0, 3, 2, 1])
        XWKT = tf.transpose(XWK, perm=[0, 3, 1, 2])
        #TRANSPOSE HERE FOR MORE STABILITY
        A_pre=tf.matmul(tf.matmul(XWKt, XWQt), XWKT)
        A= SoftPDmax_additiveScale_N_c_d_d(A_pre)
        AXA = tf.matmul(tf.matmul(A, X), A,transpose_b=True)
        res = tf.multiply(AXA, self.llambda[None,:,None,None])
        res = inputs + res
        return res

    def get_config(self):
        return {}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class layer_N_C_d_d_spd_activation_scaled(tf.keras.layers.Layer):
    """
    A layer that applies SPD activation function to the input tensors.

    Args:
        N_exp (int): Number of exponentiation terms in the activation function.
    """
    def __init__(self, N_exp=3, **kwargs):
        self.N_exp = N_exp
        super(layer_N_C_d_d_spd_activation_scaled, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_units = int(input_shape[1])

        def init1(shape,dtype=ddtype):
            return tf.cast(tf.ones(shape, dtype=dtype) / self.N_exp, dtype)

        self.a = self.add_weight(
            shape=(self.N_exp,),
            initializer=init1,
            trainable=True,
            dtype=ddtype,
            name="a",
            constraint=l1_constraintLessEqual
        )
        self.llambda = self.add_weight(
            shape=(self.input_units,),
            initializer=lam_init_eps(lam_initial),
            trainable=True,
            dtype=ddtype,
            name="llambda",
            constraint=tf.keras.constraints.NonNeg()
        )
        self.llambda_self = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(value=0.99),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.8, max_value=1.1),
            dtype=ddtype,
            name="llambda_self"
        )

    def call(self, inputs, **kwargs):
        dia = tf.abs(tf.linalg.diag_part(inputs))
        dia = tf.maximum(dia, tf.expand_dims(tf.reduce_sum(dia, axis=2), axis=2) / 100000)
        dia = tf.maximum(dia, 0.0001)
        di = tf.sqrt(dia)
        diagonal_part = tf.linalg.diag(di)
        d_inv = tf.linalg.diag(1 / di)
        cors = tf.matmul(tf.matmul(d_inv, inputs), d_inv)
        cors = tf.math.minimum(cors, 1)
        cors = tf.math.maximum(cors, -1)
        ##
        # @tf.function(reduce_retracing=True)
        def body(i, res_exp_powers, summ):
            res_exp_powers *= cors
            summ += (self.a[i] * res_exp_powers)
            return i + 1, res_exp_powers, summ

        i = tf.constant(1)
        res_exp_powers = cors
        summ = res_exp_powers * self.a[0]
        N_exp=self.N_exp
        def while_cond(i, res_exp_powers, summ):
            return i < N_exp

        _, _, res = tf.while_loop(
            while_cond,
            body,
            [i, res_exp_powers, summ]
        )
        res = tf.multiply(inputs, self.llambda_self) + tf.multiply(res, self.llambda[None, :, None, None])
        return res

    def get_config(self):
        return {"N_exp": self.N_exp}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @tf.function(reduce_retracing=True)
def data_N_M_d_c_to_cov_N_c_d_d(inputs):
    """
        Convert input tensors of shape (N, M, d, C) to covariance matrices of shape (N, C, d, d).

        Args:
            inputs (tf.Tensor): Input tensor of shape (N, M, d, C).

        Returns:
            tf.Tensor: Covariance matrices of shape (N, C, d, d).
    """
    # Transpose the last two dimensions of x to (N, M, C, d)
    inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
    # Calculate the mean of x along the second dimension
    mean = tf.reduce_mean(inputs, axis=2, keepdims=True)
    # Subtract the mean from x
    centered_x = inputs - mean
    # Calculate the covariance matrices
    cov_matrices = tf.matmul(centered_x, centered_x, transpose_a=True) / tf.cast(tf.shape(inputs)[2], tf.float32)
    return cov_matrices


def frob(W):
    """
        Compute the Frobenius norm of a tensor.

        Args:
            W (tf.Tensor): Input tensor.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(W)))


class layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2(tf.keras.layers.Layer):
    """
    Applies LogEig layer, calculates probabilities for the 3 output classes
    """
    def __init__(self, **kwargs):
        super(layer_N_c_d_d_to_N_d_d_3_LogEig_softmax2, self).__init__(**kwargs)
        self.ln_em=tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        neurons_in = tf.cast(input_shape[0][1],tf.int32)+10
        self.w = self.add_weight(
            shape=(neurons_in, 3),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            #constraint=tf.keras.constraints.NonNeg(),
            dtype=tf.float32,
            name='w'
        )
        self.w2 = self.add_weight(
            shape=(neurons_in, neurons_in),
            initializer=tf.keras.initializers.HeNormal(),
            #initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=self.dtype,
            name="w2"
        )
        self.b2 = self.add_weight(
            shape=(neurons_in,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=self.dtype,
            name="b2"
        )
        self.w3 = self.add_weight(
            shape=(neurons_in, neurons_in),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=self.dtype,
            name="w3"
        )
        self.b3 = self.add_weight(
            shape=(neurons_in,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=self.dtype,
            name="b3"
        )
        self.w4 = self.add_weight(
            shape=(neurons_in, neurons_in),
            initializer=tf.keras.initializers.HeNormal(),
            #initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=self.dtype,
            name="w4"
        )
        self.b4 = self.add_weight(
            shape=(neurons_in,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=self.dtype,
            name="b4"
        )
        self.w5 = self.add_weight(
            shape=(neurons_in, neurons_in),
            #initializer=tf.keras.initializers.HeNormal(),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            dtype=self.dtype,
            name="w4"
        )
        self.b5 = self.add_weight(
            shape=(neurons_in,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=self.dtype,
            name="b4"
        )

        self.w_embedding_1 = self.add_weight(
            shape=(2, 10),
            initializer=tf.keras.initializers.GlorotNormal(),
            #initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=self.dtype,
            name="w_embedding_1"
        )
        self.b_embedding_1 = self.add_weight(
            shape=(10,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=self.dtype,
            name="b_embedding_1"
        )

        self.w_embedding_2 = self.add_weight(
            shape=(10, 100),
            #initializer=tf.keras.initializers.GlorotNormal(),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=self.dtype,
            name="w_embedding_2"
        )
        self.b_embedding_2 = self.add_weight(
            shape=(100,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=self.dtype,
            name="b_embedding_2"
        )
        self.w_embedding_3 = self.add_weight(
            shape=(100, 10),
            initializer=tf.keras.initializers.GlorotNormal(),
            #initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            dtype=self.dtype,
            name="w_embedding_2"
        )
        self.b_embedding_3 = self.add_weight(
            shape=(10,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=self.dtype,
            name="b_embedding_2"
        )

    # @tf.function(reduce_retracing=True)
    def call(self, inputs, **kwargs):
        N = tf.cast(tf.shape(inputs[0])[0],tf.float32)
        M = tf.cast(inputs[1], tf.float32)
        d=tf.cast(tf.shape(inputs[0])[2],tf.float32)

        M_d_matrix = tf.stack([M/500, d/100], axis=-1)[None,]

        em=tf.matmul(M_d_matrix,self.w_embedding_1)+self.b_embedding_1
        em2=tf.nn.relu(tf.matmul(self.ln_em(em),self.w_embedding_2)+self.b_embedding_2)
        em3=em+tf.matmul(em2,self.w_embedding_3)+self.b_embedding_3

        mat=tf.ones((N,d,d,1))
        embeddings=mat * em3


        ip=inputs[0]

        eigvals, eigvecs = tf.linalg.eigh(0.5*(ip+tf.transpose(ip,[0,1,3,2])))
        log_eigvals = tf.math.log(tf.maximum(eigvals, 0.0001))

        # Multiply the eigenvectors by the log of eigenvalues
        log_eigvecs = tf.matmul(eigvecs, tf.linalg.diag(log_eigvals))
        log_inputs = tf.matmul(log_eigvecs, tf.transpose(eigvecs,(0,1,3,2)))
        log_inputs_T=tf.transpose(log_inputs,(0,2,3,1))

        euklidean_inputs = tf.concat([log_inputs_T, embeddings], axis=-1)

        res_h=tf.nn.relu(tf.matmul(matrixNormalization_N_d_d_c(euklidean_inputs),self.w2)+self.b2)


        h2=euklidean_inputs+tf.matmul(res_h,self.w3)+self.b3

        res_h2 = tf.nn.relu(tf.matmul(matrixNormalization_N_d_d_c(h2), self.w4) + self.b4)

        h3 = h2+tf.matmul(res_h2, self.w5) + self.b5
        probs=tf.nn.softmax(tf.matmul(h3,self.w),axis=3)
        return probs

    def get_config(self):
        return {}

    def compute_output_shape(self, input_shape):
        return tf.concat([input_shape[0], input_shape[2], input_shape[3], 3], 0)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class layer_channels_dense_res_N_M_d_c(tf.keras.layers.Layer):
    """Dense CxC layer in the channels - added normalization"""
    def __init__(self, inner_channels, **kwargs):
        self.inner_channels=inner_channels
        super(layer_channels_dense_res_N_M_d_c, self).__init__(**kwargs)

    def build(self, input_shape):
        neurons_in = int(input_shape[3])
        self.w1 = self.add_weight(
            shape=(neurons_in, self.inner_channels),
            #initializer=tf.keras.initializers.RandomUniform(minval=0,maxval=lam_initial / tf.sqrt(tf.cast(neurons_in, tf.float32)) * 2),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True,
            #constraint=tf.keras.constraints.NonNeg(),
            dtype=tf.float32,
            name='w1'
        )
        self.w2 = self.add_weight(
            shape=(self.inner_channels, neurons_in),
            #initializer=tf.keras.initializers.RandomUniform(minval=0,maxval=lam_initial / tf.sqrt(tf.cast(neurons_in, tf.float32)) * 2),
            initializer=tf.keras.initializers.HeNormal(),
            #initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            #constraint=tf.keras.constraints.NonNeg(),
            dtype=tf.float32,
            name='w2'
        )
        self.llambda = self.add_weight(
            shape=(neurons_in,),
            initializer=lam_init_eps(lam_initial),
            trainable=True,
            dtype=ddtype,
            name='llambda'
        )

    # @tf.function(reduce_retracing=True)
    def call(self, inputs, **kwargs):
        inputs_w = tf.matmul(tf.nn.relu(tf.matmul(observationalNormalization_N_M_d_c(inputs), self.w1)), self.w2)
        res = inputs + tf.multiply(inputs_w,self.llambda)
        return res

    def get_config(self):
        return {'inner_channels':self.inner_channels}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def lam_init_eps(eps=10 ** -1):
    def initializer(shape, dtype=tf.float32):
        return tf.repeat(tf.cast(eps, dtype), shape)
    return initializer


def SoftPDmax_additiveScale_N_c_d_d(cov_xx):
    '''Custom Softmax function'''
    cov_xx = cov_xx - tf.reduce_max(cov_xx,axis=(2,3),keepdims=True) #just for computational stability
    cov_xx=tf.exp(cov_xx)
    dia = tf.reduce_sum(cov_xx,-1)
    dia = tf.maximum(dia, tf.expand_dims(tf.reduce_sum(dia, axis=2), axis=2) / 100000)
    dia = tf.maximum(dia, 0.00001)
    di = tf.sqrt(dia)
    d_inv = tf.linalg.diag(1 / di)
    cor = tf.matmul(tf.matmul(d_inv, cov_xx), d_inv)
    return cor


# @tf.function(reduce_retracing=True)
def l1_constraintLessEqual(w):
    w_g0 = w * tf.keras.backend.cast(tf.keras.backend.greater_equal(w, 0), tf.keras.backend.floatx()) \
           * tf.keras.backend.cast(tf.keras.backend.less_equal(tf.keras.backend.sum(w), 1), tf.keras.backend.floatx())
    return w_g0


# @tf.function(reduce_retracing=True)
def l1_constraint_columns(w):
    w_g0 = tf.math.maximum(w, 0)
    w_g0 = w_g0 / tf.math.reduce_sum(w_g0, axis=0)
    return w_g0


class MultiHeadAttention_N_M_d_C_Feature(tf.keras.Model):
    """
        Multi-head attention layer for the feature dimension.
        Initializes and calls attention between features layer

        Args:
            num_heads (int): Number of attention heads.
            inner_channels (int): Number of inner channels.
            activation (str): Activation function. Default is "relu".
            **kwargs: Additional keyword arguments.
    """
    def __init__(self, num_heads,inner_channels, activation="relu", **kwargs):
        super(MultiHeadAttention_N_M_d_C_Feature, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_units = int(inner_channels / num_heads)
        self.inner_channels=inner_channels
        self.activation=activation
        for l in range(1, num_heads+1):
            setattr(self, f"head_{l}",
                    layer_N_M_d_C_attention_features_for_each_sample(self.head_units, activation=activation))

    def call(self, inputs, **kwargs):
        attention_heads = tf.split(inputs, self.num_heads, axis=-1)
        heads = [getattr(self, f"head_{l + 1}")(attention_heads[l]) for l in range(self.num_heads)]
        outputs = tf.concat(heads, axis=-1)
        return outputs

    def get_config(self):
        return {'num_heads': self.num_heads,
                'inner_channels': self.inner_channels,
                'activation':self.activation}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)


class MultiHeadAttention_N_M_d_C_Sample(tf.keras.Model):
    """
        Multi-head attention layer for the sample dimension.
        Initializes and calls attention between samples layer

        Args:
            num_heads (int): Number of attention heads.
            inner_channels (int): Number of inner channels.
            activation (str): Activation function. Default is "relu".
            **kwargs: Additional keyword arguments.
    """

    def __init__(self, num_heads,inner_channels, activation="relu", **kwargs):
        super(MultiHeadAttention_N_M_d_C_Sample, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_units = int(inner_channels / num_heads)
        self.inner_channels=inner_channels
        self.activation=activation
        for l in range(1, num_heads+1):
            setattr(self, f"head_{l}",
                    layer_N_M_d_C_attention_samples_for_each_feature(self.head_units, activation=activation))

    def call(self, inputs, **kwargs):
        attention_heads = tf.split(inputs, self.num_heads, axis=-1)
        heads = [getattr(self, f"head_{l + 1}")(attention_heads[l]) for l in range(self.num_heads)]
        outputs = tf.concat(heads, axis=-1)

        return outputs

    def get_config(self):
        return {'num_heads': self.num_heads,
                'inner_channels': self.inner_channels,
                'activation':self.activation}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)


class MultiHeadAttention_N_C_d_d_bilinear(tf.keras.Model):
    """
        Multi-head attention layer for bilinear attention.
        Initializes and calls bilinear attention on several heads

        Args:
            num_heads (int): Number of attention heads.
            inner_channels (int): Number of inner channels.
            activation (str): Activation function. Default is "relu".
            **kwargs: Additional keyword arguments.
    """

    def __init__(self, num_heads, **kwargs):
        super(MultiHeadAttention_N_C_d_d_bilinear, self).__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.head_units = int(input_shape[1] / self.num_heads)
        for l in range(1, self.num_heads+1):
            setattr(self, f"head_{l}",
                    layer_N_C_d_d_bilinear_attention_cov2cor_spd())

    def call(self, inputs, **kwargs):
        attention_heads = tf.split(inputs, self.num_heads, axis=1)
        heads = [getattr(self, f"head_{l + 1}")(attention_heads[l]) for l in range(self.num_heads)]
        outputs = tf.concat(heads, axis=1)
        return outputs

    def get_config(self):
        return {'num_heads': self.num_heads}

    def compute_output_shape(self, input_shape):
        return input_shape

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(**config)



def print_symmetry_N_c_d_d(mat,comment=""):
    tf.print(comment)
    tf.print(tf.reduce_sum(tf.abs(mat - tf.transpose(mat, perm=[0, 1, 3, 2]))))
    tf.print(tf.reduce_max(tf.abs(mat - tf.transpose(mat, perm=[0, 1, 3, 2]))))

def print_checkNan(mat,comment="Nan values: "):
    tensor=tf.reduce_sum(tf.cast(tf.math.is_nan(mat),tf.float32))
    tf.print(comment,tensor)

def print_variance_N_c_d_d_over_d_d(mat,comment):
    tf.print("Variance inputs before "+comment+" Min:", tf.reduce_min(tf.math.reduce_variance(mat, axis=[2, 3])))
    tf.print("Variance inputs before "+comment+" Max:", tf.reduce_max(tf.math.reduce_variance(mat, axis=[2, 3])))
    tf.print("Variance inputs before "+comment+" Mean:", tf.reduce_mean(tf.math.reduce_variance(mat, axis=[2, 3])))

def print_variance_N_d_d_c_over_d_d(mat,comment):
    tf.print("Variance inputs before "+comment+" Min:", tf.reduce_min(tf.math.reduce_variance(mat, axis=[1, 2])))
    tf.print("Variance inputs before "+comment+" Max:", tf.reduce_max(tf.math.reduce_variance(mat, axis=[1, 2])))
    tf.print("Variance inputs before "+comment+" Mean:", tf.reduce_mean(tf.math.reduce_variance(mat, axis=[1, 2])))


def print_statistics_N_c_d(mat,comment):
    tf.print("Min Min"+comment+": ", tf.reduce_min(mat, axis=[1,2]))
    tf.print("Max Max"+comment+": ", tf.reduce_max(mat, axis=[1,2]))
    tf.print("Mean Mean"+comment+": ", tf.reduce_mean(mat, axis=[1,2]))


def print_statistics_N_d_d_c_over_d_d(mat,comment):
    tf.print("Min "+comment+" Min:", tf.reduce_min(tf.math.reduce_min(mat, axis=[1, 2])))
    tf.print("Max "+comment+" Max:", tf.reduce_max(tf.math.reduce_max(mat, axis=[1, 2])))
    tf.print("Mean "+comment+" Mean:", tf.reduce_mean(tf.math.reduce_mean(mat, axis=[1, 2])))


def matrixNormalization_N_d_d_c(mat):
    mat_c=mat-tf.reduce_mean(mat,[1,2],keepdims=True)
    mat_s=mat_c/tf.maximum(tf.math.reduce_std(mat_c,[1,2],keepdims=True),0.001)
    return mat_s


def print_statistics_N_d_d_c_for_each_c(mat):
    mins=tf.math.reduce_min(mat, axis=[1, 2])
    maxs= tf.math.reduce_max(mat, axis=[1, 2])
    means= tf.math.reduce_mean(mat, axis=[1, 2])
    for c in range(tf.shape(mat)[3]):
        tf.print(c," Min: ",mins[:,c])
        tf.print(c," Max: ",maxs[:,c])
        tf.print(c," Mean: ",means[:,c])

def observationalNormalization_N_M_d_c(mat):
    mat_c=mat-tf.reduce_mean(mat,[1],keepdims=True)
    mat_s=mat_c/tf.maximum(tf.math.reduce_std(mat_c,[1],keepdims=True),0.001)
    return mat_s