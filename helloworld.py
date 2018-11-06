import tensorflow as tf

h = tf.constant("gello")
w = tf.constant(" world!")
hw = h + w

with tf.Session() as sess:
    ans = sess.run(hw)

print ans