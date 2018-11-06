import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = 'mnist'
steps = 200000
b_size = 50
show_size = 50

data = input_data.read_data_sets(DATA_DIR, one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
e_w1 = tf.Variable(tf.random_normal([784, 256]))
e_b1 = tf.Variable(tf.random_normal([b_size, 256]))
e_w2 = tf.Variable(tf.random_normal([256, 128]))
e_b2 = tf.Variable(tf.random_normal([b_size, 128]))

d_w1 = tf.Variable(tf.random_normal([128, 256]))
d_b1 = tf.Variable(tf.random_normal([b_size, 256]))
d_w2 = tf.Variable(tf.random_normal([256, 784]))
d_b2 = tf.Variable(tf.random_normal([b_size, 784]))


def encoder(data):
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

