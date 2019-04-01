

import numpy as np
import tensorflow as tf

from ops import build_fnn, dcgan_decode, dcgan_encode
from ops import loss_rec_normal, loss_reg_normal, loss_kl_normal


class GenMap(object):
    """Joint Conditional Variational Autoencoder"""
    def __init__(self, arch_encoder_attribute, image_size, dim_features=512, num_conv_layer=4,
                 transfer_fct=tf.nn.relu, learning_rate=1e-3, batch_size=100, rec_log_var=0.,
                 training=False, reconstruct_size=(64, 64, 3)):
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.dim_attribute = arch_encoder_attribute[0]
        self.latent_dim = arch_encoder_attribute[-1]
        self.reconstruction_dim = np.prod(reconstruct_size)

        self.x = tf.placeholder(tf.float32, [None, self.dim_attribute], name="pose_input")
        y_shape = [None]
        y_shape.extend(image_size)
        self.y = tf.placeholder(tf.float32, y_shape, name="image_input")
        self.y_small = tf.placeholder(tf.float32, [None, self.reconstruction_dim], name="image_reconstruct")

        # for VAE, internal representation has 2x params
        # for JCVAE, we have two encoders, and a single decoder for y
        arch_encoder_x = arch_encoder_attribute.copy()
        arch_encoder_x[-1] = arch_encoder_x[-1] * 2
        z_x_all = build_fnn(self.x, arch_encoder_x, transfer_fct, scope="encoder_x")
        self.z_x_mean, z_x_log_sigma_sq = tf.split(z_x_all, num_or_size_splits=2, axis=1)

        self.z_x_log_sigma_sq = tf.zeros_like(z_x_log_sigma_sq)  # set the z_x_sigma to be 1

        # reparameterization
        epsilon_x = tf.random_normal((self.batch_size, self.latent_dim))
        self.z_x = self.z_x_mean + (tf.sqrt(tf.exp(self.z_x_log_sigma_sq)) * epsilon_x)

        self.z_y_mean, self.z_y_log_sigma_sq = dcgan_encode(final_channel=dim_features,
                                                            num_layer=num_conv_layer,
                                                            inputs=self.y,
                                                            output_size=self.latent_dim,
                                                            activation=transfer_fct,
                                                            scope="encoder_y",
                                                            training=training)

        # reparameterization
        epsilon_y = tf.random_normal((self.batch_size, self.latent_dim))
        self.z_y = self.z_y_mean + (tf.sqrt(tf.exp(self.z_y_log_sigma_sq)) * epsilon_y)

        # decode
        y_rec_pre = dcgan_decode(inputs=self.z_y,
                                 output_size=reconstruct_size,
                                 num_layer=num_conv_layer,
                                 channel_start=dim_features,
                                 activation=transfer_fct,
                                 scope="decoder_y",
                                 training=training)

        self.alpha = tf.get_variable("image_variance_log", initializer=rec_log_var * tf.ones((1,)), trainable=False)

        self.y_rec = (1 + 2e-6) * tf.sigmoid(y_rec_pre) - 1e-6  # to avoid numeric issues

        self._minimize_loss(self.y_small, self.y_rec, self.z_x_mean, self.z_x_log_sigma_sq,
                            self.z_y_mean, self.z_y_log_sigma_sq, learning_rate, self.alpha)

        self.saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _minimize_loss(self, val, val_rec, z_s_mu, z_s_log_var, z_val_mu, z_val_log_var, learning_rate, val_log_var):
        """Minimize the loss of VAE, which contains kl-divergence term and reconstruction term."""
        loss_rec = loss_rec_normal(val, val_rec, log_var=val_log_var)
        loss_kl = loss_kl_normal(z_val_mu, z_val_log_var, z_s_mu, z_s_log_var)
        self.cost_1 = tf.reduce_mean(loss_rec / self.reconstruction_dim)
        self.cost_2 = tf.reduce_mean(loss_kl / self.latent_dim)
        self.cost = self.cost_1 + self.cost_2
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def batch_train(self, x, y, y_rec):
        _, cost, cost1, cost2 = self.sess.run((self.train_op, self.cost, self.cost_1, self.cost_2),
                                              feed_dict={self.x: x, self.y: y, self.y_small: y_rec})
        return cost, cost1, cost2

    def infer_by_x(self, x, sample=False):
        """Infer the latent variable z given input x"""
        if not sample:
            return self.sess.run(self.z_x_mean, feed_dict={self.x: x})
        else:
            return self.sess.run(self.z_x, feed_dict={self.x: x})

    def cov_infer_by_y(self, y):
        log_var = self.sess.run(self.z_y_log_sigma_sq, feed_dict={self.y: y})
        cov = np.diag(np.squeeze(np.exp(log_var)))
        return cov

    def cov_infer_by_x(self, x):
        log_var = self.sess.run(self.z_x_log_sigma_sq, feed_dict={self.x: x})
        cov = np.diag(np.squeeze(np.exp(log_var)))
        return cov

    def sigma_infer_by_y(self, y):
        log_var = self.sess.run(self.z_y_log_sigma_sq, feed_dict={self.y: y})
        return np.sqrt(np.exp(log_var))

    def sigma_infer_by_x(self, x):
        log_var = self.sess.run(self.z_x_log_sigma_sq, feed_dict={self.x: x})
        return np.sqrt(np.exp(log_var))

    def infer_by_y(self, y, sample=False):
        """Infer the latent variable z given label y"""
        if not sample:
            return self.sess.run(self.z_y_mean, feed_dict={self.y: y})
        else:
            return self.sess.run(self.z_y, feed_dict={self.y: y})

    def construct_y_from_x(self, x):
        z = self.sess.run(self.z_x_mean, feed_dict={self.x: x})
        y = self.sess.run(self.y_rec, feed_dict={self.z_y: z})
        return y

    def reconstruct(self, y):
        z = self.sess.run(self.z_y_mean, feed_dict={self.y: y})
        y = self.sess.run(self.y_rec, feed_dict={self.z_y: z})
        return y

    def save(self, save_path):
        self.saver.save(self.sess, save_path)
        print("Model saved in path: {}".format(save_path), flush=True)

    def restore(self, path):
        # reinitialize adam
        # self.sess.run([v.initializer for v in self.adam_params])
        if not path.endswith(".ckpt"):
            path = tf.train.latest_checkpoint(path)
        self.saver.restore(self.sess, path)
        print("Model restored from path: {}".format(path))

    def destruct(self):
        tf.reset_default_graph()
        self.sess.close()
