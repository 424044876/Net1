import tensorflow as tf

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a, b)
e = tf.add(c, d)
f = tf.subtract(d, e)


with tf.Session() as sess:
    fetches = [a, b, c, d, e, f]
    outs = sess.run(fetches)

print("out = {}".format(outs))
print(type(outs))