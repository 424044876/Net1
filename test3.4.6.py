import tensorflow as tf


x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
