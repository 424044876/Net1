import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0])
x = tf.reduce_mean(a)

with tf.Session() as sess:
    print sess.run(x)