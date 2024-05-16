import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .networks import dense_nn, cond_dense_nn

class CondVAE(object):
    def __init__(self, hps, name="cvae"):
        self.hps = hps
        self.name = name

    def enc(self, x, cond=None):
        '''
        x: [B, C]
        cond: [B, C]
        '''
        B,C = tf.shape(input=x)[0], tf.shape(input=x)[1]
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            prior_dist = tfd.MultivariateNormalDiag(tf.zeros(self.hps['hid_dimensions']),tf.ones(self.hps['hid_dimensions']))
            if cond is None:
                x = dense_nn(x, self.hps['enc_dense_hids'], 2 * self.hps['hid_dimensions'], False, "enc")
            else:
                x = cond_dense_nn(x, cond, self.hps['enc_dense_hids'], 2 * self.hps['hid_dimensions'], False, "enc")
            m, s = x[:, :self.hps['hid_dimensions']], tf.nn.softplus(x[:, self.hps['hid_dimensions']:])
            posterior_dist = tfd.MultivariateNormalDiag(m,s)
            #kl = 0.5 * tf.reduce_sum(s + m ** 2 - 1.0 - tf.log(s), axis=-1)
            kl = - tfd.kl_divergence(posterior_dist, prior_dist)
            eps = prior_dist.sample(B)
            posterior_sample = m + eps * s
        return kl, posterior_sample
    
    def dec(self, x, cond=None):
        '''
        x: [B, C]
        '''
        B,C = tf.shape(input=x)[0], tf.shape(input=x)[1]
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            if cond is None:
                x = dense_nn(x, self.hps['dec_dense_hids'], 2 * self.hps['dimension'], False, "dec")
            else:
                x = cond_dense_nn(x, cond, self.hps['dec_dense_hids'], 2 * self.hps['dimension'], False, "dec")
            m, s = x[:, :self.hps['dimension']], tf.nn.softplus(x[:, self.hps['dimension']:])
            sample_dist = tfd.MultivariateNormalDiag(loc=m, scale_diag=s)
        return sample_dist