from jax import numpy as jnp
import numpy as np
from jax import grad, jit, pmap 
import sklearn.datasets as datasets
import time

sigmoid = lambda x: 1 / (1+jnp.exp(-x))
ce_loss = lambda y_tgts, y_pred: -jnp.mean( y_tgts * jnp.log(y_pred)\
        + (1-y_tgts) * jnp.log(1-y_pred))

def forward(x, w):

    x = jnp.tanh(jnp.matmul(x,w[0]))
    x = sigmoid(jnp.matmul(x,w[1]))

    return x

@jit
def get_loss(x, w, y_tgts):

    y_pred = forward(x, w)

    return ce_loss(y_tgts, y_pred)

get_grad = grad(get_loss, argnums=(1))
jit_grad = jit(get_grad)

if __name__ == "__main__":

    x = np.random.randn(1024,128)
    y_tgts = np.random.randint(2, size=(1024,1))

    w0 = 1e-2 * np.random.randn(128,128)
    w1 = 1e-2 * np.random.randn(128,1)
    w = [w0,w1]

    t0 = time.time()
    for ii in range(10000):

        my_grad = get_grad(x,w, y_tgts)


        for idx, grads in enumerate(my_grad):
            w[idx] -= 1e-2 * grads

        if ii % 100 == 0:
            loss = get_loss(x, w, y_tgts)
            print(loss)

    t00 = time.time()

    w0 = 1e-2 * np.random.randn(128,128)
    w1 = 1e-2 * np.random.randn(128,1)
    w = [w0,w1]

    t1 = time.time()
    jit_get_loss = jit(get_loss)
    for jj in range(10000):

        my_grad = jit_grad(x, w, y_tgts)

        for idx, grads in enumerate(my_grad):
            w[idx] -= 1e-2 * grads

        if jj % 100 == 0:
            loss = get_loss(x, w, y_tgts)
            print(loss)


    t2 = time.time()

    print("loop execution time: {:.2f} s, time with jit: {:.2f} s".format(t00-t0, t2-t1))

