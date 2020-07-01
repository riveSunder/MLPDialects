import tensorflow as tf 
import numpy as np

import time


class MiniMLP(tf.keras.Model):
    def __init__(self):
        super(MiniMLP, self).__init__()

        self.w = [tf.random.uniform(shape=(128,128), minval=-1e-2, maxval=1e-2), \
            tf.random.uniform(shape=(128,1), minval=-1e-2, maxval=1e-2)] 

    def forward(self, x):
        x = tf.tanh(tf.matmul(x, self.w[0]))
        x = tf.sigmoid(tf.matmul(x, self.w[1]))

        return x

    

if __name__ == "__main__":

    model = MiniMLP()

    x = tf.random.normal(shape=(1024,128))
    y_tgts = tf.cast(tf.random.uniform(shape=(1024,1), \
            maxval=2, minval=0, dtype=tf.int32), tf.float32)

    t0 = time.time()

    for ii in range(10000):

        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(model.w[0])
            g.watch(model.w[1])

            y_pred = model.forward(x)
            loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_tgts, y_pred), axis=0)

        dl_dw1 = g.gradient(loss, model.w[1])
        dl_dw0 = g.gradient(loss, model.w[0])

        for idx, grads in enumerate([dl_dw0, dl_dw1]):

            model.w[idx] -= 1e-3 * grads

        if ii % 100 == 0:
            print(loss)

    t1 = time.time()
    print("tensorflow time elapsed: {:.2f}".format(t1-t0))
