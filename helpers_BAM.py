import tensorflow as tf


def my_loss_categorical_N_d_d_c(y_true, y_pred):
    y_t = get_off_diag_var_size_N_d_d_c(y_true)
    y_p = get_off_diag_var_size_N_d_d_c(y_pred)
    y_p = tf.keras.backend.clip(y_p, 0, 1)
    loss = tf.keras.backend.mean(tf.keras.metrics.categorical_crossentropy(y_t, y_p))
    return loss


def my_loss_categorical_N_d_d_c_binary(y_true, y_pred):
    y_true_binary = tf.concat(
        (tf.expand_dims(y_true[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_true[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)
    y_pred_binary = tf.concat(
        (tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_pred[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)
    # result = tf.concat([X[:,:,:,0], reduced], axis=-1)
    y_t = get_off_diag_var_size_N_d_d_c(y_true_binary)
    y_p = get_off_diag_var_size_N_d_d_c(y_pred_binary)
    y_p = tf.keras.backend.clip(y_p, 0, 1)
    loss = tf.keras.backend.mean(tf.keras.metrics.categorical_crossentropy(y_t, y_p))
    return loss


def my_penalty(y_pred, lam=0.5):
    penalty = lam * tf.reduce_mean(
        tf.nn.relu(y_pred[:, :, :, 2] - tf.sqrt(tf.matmul(y_pred[:, :, :, 1], y_pred[:, :, :, 1]))))
    return penalty


def my_penalty_metric(y_true, y_pred):
    penalty = tf.reduce_mean(
        tf.nn.relu(y_pred[:, :, :, 2] - tf.sqrt(tf.matmul(y_pred[:, :, :, 1], y_pred[:, :, :, 1]))))
    return penalty


def my_loss_categorical_penalty(y_true, y_pred):
    loss = 0.5 * my_loss_categorical_N_d_d_c(y_true, y_pred) + 0.5 * my_loss_categorical_N_d_d_c_binary(y_true,
                                                                                                        y_pred) + my_penalty(
        y_pred)
    return loss


def my_loss_pure_categorical_penalty(y_true, y_pred):
    loss = my_loss_categorical_N_d_d_c(y_true, y_pred) + my_penalty(y_pred)
    return loss


def my_accuracy_categorical_N_d_d_c(y_true, y_pred):
    y_t = get_off_diag_var_size_N_d_d_c(y_true)
    y_p = get_off_diag_var_size_N_d_d_c(y_pred)
    y_p = tf.keras.backend.clip(y_p, 0, 1)
    acc = tf.keras.metrics.categorical_accuracy(y_t, y_p)
    return acc


def my_accuracy_categorical_N_d_d_c_binary(y_true, y_pred):
    y_true_binary = tf.concat(
        (tf.expand_dims(y_true[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_true[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)
    y_pred_binary = tf.concat(
        (tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_pred[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)
    y_t = get_off_diag_var_size_N_d_d_c(y_true_binary)
    y_p = get_off_diag_var_size_N_d_d_c(y_pred_binary)
    y_p = tf.keras.backend.clip(y_p, 0, 1)
    acc = tf.keras.metrics.categorical_accuracy(y_t, y_p)
    return acc


def get_off_diag_var_size_N_d_d_c(M):
    M_shape = tf.shape(M)[1]
    M_shape2 = tf.stack((M_shape, M_shape))
    ones_mat = tf.ones(M_shape2)

    mask_a = tf.linalg.band_part(ones_mat, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.linalg.band_part(ones_mat, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask

    upper_triangular = tf.boolean_mask(M, mask, axis=1)
    return upper_triangular


def recallBinary(y_true, y_pred):
    # extract the entries where the true label is 2
    y_true_binary = tf.concat(
        (tf.expand_dims(y_true[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_true[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)
    y_pred_binary = tf.concat(
        (tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_pred[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)

    y_t = get_off_diag_var_size_N_d_d_c(y_true_binary)
    y_p = get_off_diag_var_size_N_d_d_c(y_pred_binary)

    y_t = tf.argmax(y_t, -1)
    y_p = tf.argmax(y_p, -1)

    y_true_cat1 = tf.keras.backend.cast(tf.equal(y_t, 1), tf.keras.backend.floatx())
    y_pred_cat1 = tf.keras.backend.cast(tf.equal(y_p, 1), tf.keras.backend.floatx())
    cat1_correct = tf.keras.backend.sum(y_true_cat1 * y_pred_cat1)
    num_cat1 = tf.keras.backend.sum(y_true_cat1)
    if num_cat1 == 0:
        num_cat1 = 1.
        cat1_correct = 1.
    # return the quotient, making sure to handle division by zero
    return cat1_correct / num_cat1


def precisionBinary(y_true, y_pred):
    y_true_binary = tf.concat(
        (tf.expand_dims(y_true[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_true[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)
    y_pred_binary = tf.concat(
        (tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_pred[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)

    # extract the entries where the true label is 2
    y_t = get_off_diag_var_size_N_d_d_c(y_true_binary)
    y_p = get_off_diag_var_size_N_d_d_c(y_pred_binary)

    y_t = tf.argmax(y_t, -1)
    y_p = tf.argmax(y_p, -1)

    y_true_cat1 = tf.keras.backend.cast(tf.equal(y_t, 1), tf.keras.backend.floatx())
    y_pred_cat1 = tf.keras.backend.cast(tf.equal(y_p, 1), tf.keras.backend.floatx())
    cat1_correct = tf.keras.backend.sum(y_true_cat1 * y_pred_cat1)
    num_est_cat1 = tf.keras.backend.sum(y_pred_cat1)
    if num_est_cat1 == 0:
        num_est_cat1 = 1.
        cat1_correct = 1.
    return cat1_correct / num_est_cat1


auc = tf.keras.metrics.AUC()


def aucBinary(y_true, y_pred):
    y_true_binary = tf.concat(
        (tf.expand_dims(y_true[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_true[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)
    y_pred_binary = tf.concat(
        (tf.expand_dims(y_pred[:, :, :, 0], -1), tf.expand_dims(tf.reduce_sum(y_pred[:, :, :, 1:3], axis=-1), -1)),
        axis=-1)

    y_t = get_off_diag_var_size_N_d_d_c(y_true_binary)
    y_p = get_off_diag_var_size_N_d_d_c(y_pred_binary)

    y_t = y_t[:, :, 1]
    y_p = y_p[:, :, 1]

    y_t = tf.reshape(y_t, [-1])
    y_p = tf.reshape(y_p, [-1])

    if tf.reduce_sum(y_t) == 0:
        return 1.
    auc.reset_state()
    return auc(y_t, y_p)
