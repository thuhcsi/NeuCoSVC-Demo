import tensorflow as tf
tfk = tf.keras
import numpy as np

# base class
class BaseTransform(object):
    def __init__(self, hps, name='base'):
        self.name = name
        self.hps = hps

        self.build()

    def build(self):
        pass

    def forward(self, x):
        raise NotImplementedError()

    def inverse(self, z):
        raise NotImplementedError()


class Transform(BaseTransform):
    def __init__(self, hps, name='transform'):
        super(Transform, self).__init__(hps, name)

    def build(self):
        self.modules = []
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            for i, name in enumerate(self.hps.transform):
                m = TRANS[name](self.hps, f'{i}')
                self.modules.append(m)

    def forward(self, x):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            logdet = 0.
            for module in self.modules:
                x, ldet = module.forward(x)
                logdet = logdet + ldet

        return x, logdet

    def inverse(self, z):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            logdet = 0.
            for module in reversed(self.modules):
                z, ldet = module.inverse(z)
                logdet = logdet + ldet

        return z, logdet


class Reverse(BaseTransform):
    def __init__(self, hps, name):
        name = f'reverse_{name}'
        super(Reverse, self).__init__(hps, name)

    def forward(self, x):
        z = tf.reverse(x, [-1])
        ldet = 0.0

        return z, ldet

    def inverse(self, z):
        x = tf.reverse(z, [-1])
        ldet = 0.0

        return x, ldet


class LeakyReLU(BaseTransform):
    def __init__(self, hps, name):
        name = f'lrelu_{name}'
        super(LeakyReLU, self).__init__(hps, name)

    def build(self):
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            self.alpha = tf.nn.sigmoid(
                tf.compat.v1.get_variable('log_alpha', 
                                initializer=5.0, 
                                dtype=tf.float32))

    def forward(self, x):
        num_negative = tf.reduce_sum(input_tensor=tf.cast(tf.less(x, 0.0), tf.float32), axis=1)
        ldet = num_negative * tf.math.log(self.alpha)
        z = tf.maximum(x, self.alpha * x)

        return z, ldet

    def inverse(self, z):
        num_negative = tf.reduce_sum(input_tensor=tf.cast(tf.less(z, 0.0), tf.float32), axis=1)
        ldet = -1. * num_negative * tf.math.log(self.alpha)
        x = tf.minimum(z, z / self.alpha)

        return x, ldet


class Coupling(BaseTransform):
    def __init__(self, hps, name):
        name = f'coupling_{name}'
        super(Coupling, self).__init__(hps, name)

    def build(self):
        d = self.hps.dimension
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            self.net1 = tfk.Sequential(name=f'{self.name}/ms1')
            for i, h in enumerate(self.hps.coupling_hids):
                self.net1.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.net1.add(tfk.layers.Dense(d, name=f'l{i+1}', kernel_initializer=tf.compat.v1.zeros_initializer()))

            self.net2 = tfk.Sequential(name=f'{self.name}/ms2')
            for i, h in enumerate(self.hps.coupling_hids):
                self.net2.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            self.net2.add(tfk.layers.Dense(d, name=f'l{i+1}', kernel_initializer=tf.compat.v1.zeros_initializer()))

    def forward(self, x):
        B = tf.shape(input=x)[0]
        d = self.hps.dimension
        ldet = tf.zeros(B, dtype=tf.float32)
        # part 1
        inp, out = x[:,::2], x[:,1::2]
        scale, shift = tf.split(self.net1(inp), 2, axis=1)
        out = (out + shift) * tf.exp(scale)
        x = tf.reshape(tf.stack([inp,out],axis=-1), [B,d])
        ldet = ldet + tf.reduce_sum(input_tensor=scale, axis=1)
        # part 2
        out, inp = x[:,::2], x[:,1::2]
        scale, shift = tf.split(self.net2(inp), 2, axis=1)
        out = (out + shift) * tf.exp(scale)
        x = tf.reshape(tf.stack([out, inp],axis=-1), [B,d])
        ldet = ldet + tf.reduce_sum(input_tensor=scale, axis=1)

        return x, ldet

    def inverse(self, z):
        B = tf.shape(input=z)[0]
        d = self.hps.dimension
        ldet = tf.zeros(B, dtype=tf.float32)
        # part 2
        out, inp = z[:,::2], z[:,1::2]
        scale, shift = tf.split(self.net2(inp), 2, axis=1)
        out = out * tf.exp(-scale) - shift
        z = tf.reshape(tf.stack([out, inp],axis=-1), [B,d])
        ldet = ldet - tf.reduce_sum(input_tensor=scale, axis=1)
        # part 1
        inp, out = z[:,::2], z[:,1::2]
        scale, shift = tf.split(self.net1(inp), 2, axis=1)
        out = out * tf.exp(-scale) - shift
        z = tf.reshape(tf.stack([inp, out], axis=-1), [B,d])
        ldet = ldet - tf.reduce_sum(input_tensor=scale, axis=1)

        return z, ldet


class LULinear(BaseTransform):
    def __init__(self, hps, name):
        name = f'linear_{name}'
        super(LULinear, self).__init__(hps, name)

    def build(self):
        d = self.hps.dimension
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            np_w = np.eye(d).astype("float32")
            self.w = tf.compat.v1.get_variable('W', initializer=np_w)
            self.b = tf.compat.v1.get_variable('b', initializer=tf.zeros([d]))

    def get_LU(self):
        d = self.hps.dimension
        W = self.w
        U = tf.linalg.band_part(W, 0, -1)
        L = tf.eye(d) + W - U
        A = tf.matmul(L, U)

        return A, L, U

    def forward(self, x):
        A, L, U = self.get_LU()
        ldet = tf.reduce_sum(input_tensor=tf.math.log(tf.abs(tf.linalg.diag_part(U))))
        z = tf.matmul(x, A) + self.b

        return z, ldet

    def inverse(self, z):
        B = tf.shape(input=z)[0]
        A, L, U = self.get_LU()
        ldet = -1 * tf.reduce_sum(input_tensor=tf.math.log(tf.abs(tf.linalg.diag_part(U))))
        Ut = tf.tile(tf.expand_dims(tf.transpose(a=U, perm=[1, 0]), axis=0), [B,1,1])
        Lt = tf.tile(tf.expand_dims(tf.transpose(a=L, perm=[1, 0]), axis=0), [B,1,1])
        zt = tf.expand_dims(z - self.b, -1)
        sol = tf.linalg.triangular_solve(Ut, zt)
        x = tf.linalg.triangular_solve(Lt, sol, lower=False)
        x = tf.squeeze(x, axis=-1)

        return x, ldet


# register all modules
TRANS = {
    'CP': Coupling,
    'R': Reverse,
    'LR': LeakyReLU,
    'L': LULinear,
}


if __name__ == '__main__':
    from pprint import pformat
    from easydict import EasyDict as edict

    hps = edict()
    hps.dimension = 8
    hps.coupling_hids = [32,32]
    hps.transform = ['L','LR','CP','R']

    x_ph = tf.compat.v1.placeholder(tf.float32, [32,8])
    
    l1 = Transform(hps, '1')
    l2 = Transform(hps, '2')
    z, fdet1 = l1.forward(x_ph)
    z, fdet2 = l2.forward(z)
    fdet = fdet1 + fdet2

    x, bdet2 = l2.inverse(z)
    x, bdet1 = l1.inverse(x)
    bdet = bdet1 + bdet2

    err = tf.reduce_sum(input_tensor=tf.square(x_ph - x))
    det = tf.reduce_sum(input_tensor=fdet + bdet)

    loss = tf.reduce_sum(input_tensor=tf.square(z)) - fdet
    train_op = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    print('='*20)
    print('Variables:')
    print(pformat(tf.compat.v1.trainable_variables()))

    for i in range(1000):
        x_nda = np.random.randn(32,8)
        feed_dict = {x_ph:x_nda}

        res = sess.run([err,det], feed_dict)
        print(f'err:{res[0]} det:{res[1]}')
        sess.run(train_op, feed_dict)