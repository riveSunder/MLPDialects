import tensorflow as tf 
import numpy as np
from tensorflow.keras.activations import sigmoid, tanh


import time


if __name__ == "__main__":

    model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, activation="tanh"),\
            tf.keras.layers.Dense(1, activation=sigmoid)])

    model.compile(optimizer="SGD", loss="binary_crossentropy")
    x = tf.random.normal(shape=(1024,128))
    y_tgts = tf.cast(tf.random.uniform(shape=(1024,1), \
            maxval=2, minval=0, dtype=tf.int32), tf.float32)

    t0 = time.time()
    
    model.fit(x, y_tgts, batch_size=4096, epochs=10000, verbose=2)

    t1 = time.time()
    print("tensorflow time elapsed: {:.2f}".format(t1-t0))
