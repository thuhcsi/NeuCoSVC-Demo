import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from easydict import EasyDict as edict

from .base import BaseModel
from .pc_encoder import *
from .flow.transforms import Transform
from .cVAE import CondVAE

class ACSetVAE(BaseModel):
    def __init__(self, hps):
        self.prior_net = LatentEncoder(hps, name='prior')
        self.posterior_net = LatentEncoder(hps, name='posterior')
        self.cvae = CondVAE(hps.vae_params, name="cvae")
        if hps.use_peq_embed:
            self.peq_embed = SetXformer(hps)
        super(ACSetVAE, self).__init__(hps)
    
    def build_net(self):
        with tf.compat.v1.variable_scope('acset_vae', reuse=tf.compat.v1.AUTO_REUSE):
            self.x = tf.compat.v1.placeholder(tf.float32, [None, self.hps.set_size, self.hps.dimension]) #[200, 1024]
            self.b = tf.compat.v1.placeholder(tf.float32, [None, self.hps.set_size, self.hps.dimension])
            self.m = tf.compat.v1.placeholder(tf.float32, [None, self.hps.set_size, self.hps.dimension])

            # build transform
            self.transform = Transform(edict(self.hps.trans_params))

            # prior
            prior_inputs = tf.concat([self.x*self.b, self.b], axis=-1) # [200, 2048]
            prior = self.prior_net(prior_inputs) # [256]
            prior_sample = prior.sample() # [256]
            prior_sample, _ = self.transform.inverse(prior_sample) # [256]
            # peq embedding
            cm = peq_embed = None
            if self.hps.use_peq_embed:
                peq_embed = self.peq_embed(prior_inputs) # [200, 256]
                C = peq_embed.get_shape().as_list()[-1] # 256
                cm = tf.reshape(peq_embed, [-1,C]) # [256]
            
            # posterior
            posterior_inputs = tf.concat([self.x*self.m, self.m], axis=-1) # [200, 2048]
            posterior = self.posterior_net(posterior_inputs) # [256]
            posterior_sample = posterior.sample() # [256]

            # kl term
            z_sample, logdet = self.transform.forward(posterior_sample) # [256], [1]
            logp = tf.reduce_sum(input_tensor=prior.log_prob(z_sample), axis=1) + logdet # [256] -> [1]
            kl = tf.reduce_sum(input_tensor=posterior.entropy(), axis=1) + logp # [256] -> [1]

            # generator 
            x = tf.reshape(self.x, [-1,self.hps.dimension]) # [1024]
            b = tf.reshape(self.b, [-1,self.hps.dimension])
            m = tf.reshape(self.m, [-1,self.hps.dimension])
            cv = tf.reshape(tf.tile(tf.expand_dims(posterior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim]) # [256] -> [1, 256] -> [1, 200, 256] -> [256]
            if not cm is None:
                c = tf.concat([cv, cm], axis=-1) # [512]
            else:
                c = cv
            # vector-wise posterior
            vec_kl, vec_post_sample = self.cvae.enc(tf.concat([x, c], axis=-1), c) # [1], [256]
            vec_kl = tf.reshape(vec_kl, [-1, self.hps.set_size]) # [200]
            recon_dist = self.cvae.dec(tf.concat([vec_post_sample, c], axis=-1), c)
            log_likel = recon_dist.log_prob(x) # [1]
            log_likel = tf.reshape(log_likel, [-1,self.hps.set_size]) # [200]
            self.set_metric = self.set_elbo = log_likel + tf.expand_dims(kl, axis=1) / self.hps.set_size # [200]
            log_likel = tf.reduce_mean(input_tensor=log_likel, axis=1)
            tf.compat.v1.summary.scalar('log_likel', tf.reduce_mean(input_tensor=log_likel))

            # elbo
            self.elbo = log_likel + kl / self.hps.set_size + tf.reduce_mean(input_tensor=vec_kl, axis=1)
            self.metric = self.elbo
            self.loss = tf.reduce_mean(input_tensor=-self.elbo)
            tf.compat.v1.summary.scalar('loss', self.loss)

            # sample
            x = tf.reshape(self.x, [-1,self.hps.dimension])
            b = tf.reshape(self.b, [-1,self.hps.dimension])
            m = tf.reshape(self.m, [-1,self.hps.dimension])
            cv = tf.reshape(tf.tile(tf.expand_dims(prior_sample, axis=1), [1,self.hps.set_size,1]), [-1,self.hps.latent_dim])
            if not cm is None:
                c = tf.concat([cv, cm], axis=-1)
            else:
                c = cv
            vec_prior_sample = tf.random.normal(shape=tf.shape(input=vec_post_sample))
            sample_dist = self.cvae.dec(tf.concat([vec_prior_sample, c], axis=-1), c)
            log_likel = sample_dist.log_prob(x)
            sample = sample_dist.sample()
            self.sample = tf.reshape(sample, [-1, self.hps.set_size, self.hps.dimension])

            # compress

            self.log_likel = tf.reshape(log_likel, [-1,self.hps.set_size]) + vec_kl
        
        # total_params = 0
        # for variable in tf.trainable_variables():
        #     shape = variable.get_shape()
        #     params = 1
        #     for dim in shape:
        #         params *= dim
        #     total_params += params

        # print("Total number of parameters: ", total_params)