import tensorflow as tf


def get_weight(shape, lam):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lam)(var)
    )
    return var

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))


