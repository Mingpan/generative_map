# Tensorflow related operations

import numpy as np
import tensorflow as tf


def build_fnn(x, units_list, activation=tf.sigmoid, scope="fnn", reuse=False):
    """Construct a fnn under the scope, if the scope already exists, reuse needs to be set as true."""
    num_layers = len(units_list)
    h = x
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(num_layers - 1):
            a = tf.layers.dense(h, units_list[i+1], activation=activation,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer())
            if i < num_layers - 2:
                h = activation(a)
            else:
                h = a
    return h


def loss_rec_bernoulli(ori, rec):
    return -tf.reduce_sum(ori * tf.log_sigmoid(rec) + (1 - ori) * tf.log_sigmoid(-rec), reduction_indices=1)


def loss_rec_categorical(ori, rec):
    """Categorial reconstrution """
    return -tf.reduce_sum(ori * tf.nn.log_softmax(rec), reduction_indices=1)


def loss_rec_normal(ori, mean, log_var=0.):
    """Negative log-likelihood of a normal distribution"""
    dim = ori.shape[1].value
    return 0.5 * (dim * np.log(np.pi * 2) + tf.reduce_sum(log_var + tf.square(ori - mean) / tf.exp(log_var),
                                                          reduction_indices=1))


def loss_reg_normal(mean, log_var):
    """(Positive) KL div from given distribution to standard normal distribution"""
    return -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), reduction_indices=1)


def loss_kl_normal(mean1, log_var1, mean2, log_var2):
    """(Positive) KL divergence from 1 to 2"""
    return 0.5 * tf.reduce_sum(-1 - log_var1 + log_var2
                               + tf.square(mean1 - mean2) / tf.exp(log_var2)
                               + tf.exp(log_var1 - log_var2),
                               reduction_indices=1)

# Conv / FSConv


def dcgan_decode(inputs, output_size, num_layer=3, channel_start=512, activation=tf.nn.relu,
                 scope="deconv", reuse=False, training=False):
    """Image decoder, taken ideas from the DC-GAN.

    :param inputs: the input tensor, 2D
    :param output_size: the size of the desired generated picture, a triple, e.g. (64, 64, 3)
    :param num_layer: number of Conv layers
    :param channel_start: number of initial features in the network
    :param activation: the activation function used
    :param scope: the tensorflow scope
    :param reuse: set to true if this scope of params were created somewhere before, and need to be reused.
    :param training: set to true if in training mode, for batch normalization only
    :returns
        h: the output tensor, 4D
    """
    y_w, y_h, y_c = output_size
    original_h = y_h // (2 ** num_layer)
    original_w = y_w // (2 ** num_layer)
    with tf.variable_scope(scope, reuse=reuse):
        # reshape layer --> cubic layer
        net = tf.layers.dense(inputs, channel_start * original_h * original_w, activation=activation,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer())
        net = tf.reshape(net, [-1, original_h, original_w, channel_start])

        for i in range(num_layer - 1):
            num_filter = channel_start // (2 ** (i+1))
            net = tf.layers.conv2d_transpose(net, filters=num_filter, kernel_size=[5, 5],
                                             strides=2, padding='same')
            # net = tf.layers.batch_normalization(net, training=training)
            net = activation(net)

        net = tf.layers.conv2d_transpose(net, filters=y_c, kernel_size=[5, 5], strides=(2, 2),
                                         padding='same')

        h = tf.reshape(net, [-1, np.prod(output_size)])

    return h


def dcgan_encode(inputs, output_size, num_layer=3, final_channel=512, activation=tf.nn.relu,
                 scope="encode", reuse=False, training=False):
    """Image encoder, taken ideas from the DC-GAN.

    :param inputs: the input tensor, should be 4D.
    :param output_size: the output size of the encoder, one dimensional, identical for mean and log variance
    :param num_layer: how many conv layers are stacked, excluding the final linear layer
    :param final_channel: number of filters for the final conv layer, also defines the sizes for previous layers
    :param activation: activation function in use for conv layers, excluding the final linear layer
    :param scope: scope name for tensorflow graph
    :param reuse: if reuse parameters, set to true if this scope was created somewhere before and needs to be reused
    :param training: if is in the training mode, for batch norm
    :returns
        mean: a tensor of size [inputs.shape[0], output_size]
        log_variance: a tensor of size [inputs.shape[0], output_size]
    """
    with tf.variable_scope(scope, reuse=reuse):

        net = inputs

        for i in range(num_layer):
            scale = 2**(num_layer - i - 1)  # e.g. 3-layer: 2**2, 2**1, 2**0
            net = tf.layers.conv2d(net, filters=final_channel / scale, kernel_size=[5, 5],
                                   strides=2, padding='same')
            # net = tf.layers.batch_normalization(net, training=training)
            net = activation(net)

        net = tf.reduce_mean(net, [1, 2], keepdims=False)

        mean = tf.layers.dense(inputs=net, units=output_size,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               bias_initializer=tf.zeros_initializer())
        log_variance = tf.layers.dense(inputs=net, units=output_size,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       bias_initializer=tf.zeros_initializer())
    return mean, log_variance