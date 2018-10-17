import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

def tvar(shape, name):
	return tf.get_variable(
        name, shape=shape, dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer()
    )

def dmap(di, dm, do, name):
    return [
        tvar([dm,di], 'k_%s'%name),
        tvar([dm,do], 'v_%s'%name)
    ]

def similarity(q, k, divisor='norm'):
    numerator = tf.matmul(q, tf.transpose(k))
    denomerator = 1
    q_norm = tf.norm(q, axis=-1, keepdims=True)
    k_norm = tf.norm(k, axis=-1, keepdims=True)
    norm_prod = tf.matmul(q_norm, tf.transpose(k_norm)) + 1e-8
    if divisor == 'norm':
        denomerator = norm_prod
    if divisor == 'sqrt':
        denomerator = tf.sqrt(norm_prod)
    return tf.divide(numerator, denomerator)

def wgen(x, dm, Wc=None, divisor='norm'):
    if Wc is not None:
        x = tf.matmul(x, Wc)
    k, v = dm
    a = similarity(x, k, divisor)
    W = tf.matmul(a, v)
    return W

def wgyn(x, dm, Wc=None, divisor='norm'):
    W = wgen(x, dm, Wc, divisor)
    h = tf.matmul(x, W)
    return h
    
def dense(x, bW, activation=None):
    x = tf.concat([x, tf.zeros(list(x.shape[:-1])+[1])], -1)
    h = tf.matmul(x, bW)
    if activation is not None:
        h = activation(h)
    return h