from autograd import numpy as np
from autograd import grad
import time

sigmoid = lambda x: 1 / (1+np.exp(-x))
ce_loss = lambda y_tgts, y_pred: -np.mean( y_tgts * np.log(y_pred)\
        + (1-y_tgts) * np.log(1-y_pred))

def forward(x, w):

    x = np.tanh(np.matmul(x,w[0]))
    x = sigmoid(np.matmul(x,w[1]))

    return x

def get_loss(x, w, y_tgts):

    y_pred = forward(x, w)

    return ce_loss(y_tgts, y_pred)

get_grad = grad(get_loss, argnum=(1))

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

    print("loop time (autograd): {:.2f} s".format(t00-t0))

    
