import numpy as np
import tensorflow as tf
initializer = tf.compat.v1.random_normal_initializer(stddev=0.001)
def dense_nn(x, dims, dim_out, norm=True, name='dense_nn'):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        for i, size in enumerate(dims):
            x = tf.compat.v1.layers.dense(x, size, name=f'd{i}', kernel_initializer=initializer)
            if norm:
                x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.leaky_relu(x)
        x = tf.compat.v1.layers.dense(x, dim_out, name='d_out', kernel_initializer=initializer)

        return x
    
def cond_dense_nn(x, cond, dims, dim_out, norm=True, name='cond_dense_nn'):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        for i, size in enumerate(dims):
            x = tf.compat.v1.layers.dense(x, size, name=f'd{i}', kernel_initializer=initializer)
            if norm:
                x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.leaky_relu(x)
            c = tf.compat.v1.layers.dense(cond, size, name=f'c{i}', kernel_initializer=initializer)
            x = tf.sigmoid(c) * x
        x = tf.compat.v1.layers.dense(x, dim_out, name='d_out', kernel_initializer=initializer)
        return x
    
def large_cond_dense_nn(x, cond, dims, dim_out, norm=True, name='cond_dense_nn'):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        for i, size in enumerate(dims):
            x = tf.compat.v1.layers.dense(x, size, name=f'd{i}', kernel_initializer=initializer)
            if norm:
                x = tf.contrib.layers.layer_norm(x)
            x = tf.nn.leaky_relu(x)
            c = dense_nn(cond, [256,256], size, False, name=f'c{i}')
            x = tf.sigmoid(c) * x
        x = tf.compat.v1.layers.dense(x, dim_out, name='d_out')
        return x
    
def res_block(input, dim, block_name):
    x = tf.compat.v1.layers.dense(input, dim, name=f'{block_name}_1', kernel_initializer=initializer)
    x = tf.nn.leaky_relu(x)
    x = tf.compat.v1.layers.dense(input, dim, name=f'{block_name}_2', kernel_initializer=initializer)
    x += input
    x = tf.nn.leaky_relu(x)
    return x

    
def cond_resnet(x, cond, dims, dim_out, norm=True, name='cond_resnet'):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        x = tf.compat.v1.layers.dense(x, dims[0], name='1', kernel_initializer=initializer)
        for i, size in enumerate(dims):
            x = res_block(x, size, block_name=f'res_block_{i}')
            if norm:
                x = tf.contrib.layers.layer_norm(x)
            c = dense_nn(cond, [256,256], size, False, name=f'c{i}')
            x = tf.sigmoid(c) * x
        x = tf.compat.v1.layers.dense(x, dim_out, name='d_out', kernel_initializer=initializer)
        return x


def convnet(x, dims, dim_out, name='convnet'):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        for i, d in enumerate(dims):
            x = tf.compat.v1.layers.conv2d(x, d, 3, padding='same', name=f'c{i}_1')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.compat.v1.layers.conv2d(x, d, 3, padding='same', name=f'c{i}_2')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.compat.v1.layers.max_pooling2d(x, 2, 2)
        x = tf.compat.v1.layers.flatten(x)
        x = tf.compat.v1.layers.dense(x, dim_out, name='d1')
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.leaky_relu(x)
        x = tf.compat.v1.layers.dense(x, dim_out, name='d2')

    return x

def peq_convnet(x, dims, dim_out, attention, name='peq_convnet'):
    B,N,H,W,C = tf.shape(input=x)[0], tf.shape(input=x)[1], *x.get_shape().as_list()[2:]
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        # downsample
        x = tf.reshape(x, [-1,H,W,C])
        for d in dims[:2]:
            x = tf.compat.v1.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.compat.v1.layers.conv2d(x, d, 3, strides=(2,2), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            H, W, C = H//2, W//2, d
        x = tf.reshape(x, [B,N,H,W,C])
        # attention across set dimension
        x = tf.reshape(tf.transpose(a=x, perm=[0,2,3,1,4]), [B*H*W,N,C])
        rep = attention(x, x, x)
        x += rep
        x = tf.transpose(a=tf.reshape(x, [B,H,W,N,C]), perm=[0,3,1,2,4])
        # downsample
        x = tf.reshape(x, [-1,H,W,C])
        for d in dims[2:]:
            x = tf.compat.v1.layers.conv2d(x, d, 3, strides=(1,1), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            x = tf.compat.v1.layers.conv2d(x, d, 3, strides=(2,2), padding='same')
            x = tf.contrib.layers.instance_norm(x)
            x = tf.nn.leaky_relu(x)
            H, W, C = H//2, W//2, d
        x = tf.compat.v1.layers.flatten(x)
        x = tf.compat.v1.layers.dense(x, dim_out)
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.leaky_relu(x)
        x = tf.compat.v1.layers.dense(x, dim_out)
        x = tf.reshape(x, [B,N,dim_out])

    return x

def peq_resblock(x, dim, attention, name='peq_resnet'):
    B,N,H,W,C = tf.shape(input=x)[0], tf.shape(input=x)[1], *x.get_shape().as_list()[2:]
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        res = tf.reshape(tf.transpose(a=x, perm=[0,2,3,1,4]), [B*H*W,N,C])
        res = attention(res, res, res)
        res = tf.transpose(a=tf.reshape(res, [B,H,W,N,C]), perm=[0,3,1,2,4])
        res = tf.reshape(res, [B*N,H,W,C])
        res = tf.compat.v1.layers.conv2d(res, dim, 3, strides=(1,1), padding='same')
        res = tf.contrib.layers.instance_norm(res)
        res = tf.nn.leaky_relu(res)
        res = tf.compat.v1.layers.conv2d(res, dim, 3, strides=(1,1), padding='same')
        res = tf.contrib.layers.instance_norm(res)
        res = tf.reshape(res, [B,N,H,W,C])
        x += res
        x = tf.nn.leaky_relu(x)

    return x

