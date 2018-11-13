import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'mnist'
path = 'moudel'
steps = 10000
b_size = 50
show_size = 50

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
con_kernel = tf.Variable(tf.random_normal([4, 4, 1, 1]))
e_w1 = tf.Variable(tf.random_normal([784, 256]))
e_b1 = tf.Variable(tf.random_normal([b_size, 256]))
e_w2 = tf.Variable(tf.random_normal([256, 128]))
e_b2 = tf.Variable(tf.random_normal([b_size, 128]))

d_w1 = tf.Variable(tf.random_normal([128, 256]))
d_b1 = tf.Variable(tf.random_normal([b_size, 256]))
d_w2 = tf.Variable(tf.random_normal([256, 784]))
recon_kernel = tf.Variable(tf.random_normal([4, 4, 1, 1]))
d_b2 = tf.Variable(tf.random_normal([b_size, 784]))


def encoder(data):
    i0 = tf.reshape(data, [-1, 28, 28, 1])
    tf.nn.conv2d(i0, con_kernel, strides=[1, 1, 1, 1], padding="SAME")
    i0 = tf.sigmoid(i0)
    i0 = tf.layers.flatten(i0)
    a1 = tf.add(tf.matmul(data, e_w1), e_b1)
    o1 = tf.nn.sigmoid(a1)
    a2 = tf.add(tf.matmul(o1, e_w2), e_b2)
    o2 = tf.nn.sigmoid(a2)
    return o2


def decoder(code):
    a1 = tf.add(tf.matmul(code, d_w1), d_b1)
    o1 = tf.nn.sigmoid(a1)
    a2 = tf.add(tf.matmul(o1, d_w2), d_b2)
    o2 = tf.nn.sigmoid(a2)
    o2 = tf.reshape(o2, [-1, 28, 28, 1])
    tf.nn.conv2d(o2, recon_kernel, strides=[1, 1, 1, 1], padding="SAME")
    o2 = tf.layers.flatten(o2)
    return o2


x1 = encoder(x)
x_ = decoder(x1)

loss = tf.reduce_mean(tf.square(x_ - x))

optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(steps):
        batch_xs, batch_ys = data.train.next_batch(b_size)
        c = sess.run([optimizer, loss], feed_dict={x: batch_xs[:b_size]})
        if _ % b_size == 0:
            print format(c[1])

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    res = sess.run(x_, feed_dict={x: data.test.images[:show_size]})
    for i in range(show_size):
        a[0][i].imshow(np.reshape(data.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(res[i], (28, 28)))
        plt.show()
    # se_w1 = sess.run(e_w1)
    # np.savez(os.path.join(path, 'weight'), se_w1)

# writer = tf.summary.FileWriter('TensorBoard')
# writer.add_graph(tf.get_default_graph())