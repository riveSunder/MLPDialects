import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

import time

ce_loss = lambda y_tgts, y_pred: - torch.mean(y_tgts * torch.log(y_pred)\
        + (1-y_tgts) * torch.log(1-y_pred))

class MiniMLP(nn.Module):
    def __init__(self):
        super(MiniMLP, self).__init__()

        self.w = [nn.Parameter(1e-2 * torch.randn(128,128)), \
                nn.Parameter(1e-2 * torch.randn(128,1))]

        #self.w = [torch.nn.Linear(32,32,bias=False),\
        #        torch.nn.Linear(32,1,bias=False)]

        #for idx, param in enumerate(self.w):
        #    self.add_module("layer{}".format(idx), param)

    def forward(self, x):
        x = torch.tanh(torch.matmul(x,self.w[0]))
        x = torch.sigmoid(torch.matmul(x,self.w[1]))

#        x = torch.tanh(self.w[0](x))
#        x = torch.sigmoid(self.w[1](x))

        return x

if __name__ == "__main__":

    model = MiniMLP()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    x = torch.randn(1024,128)
    y_tgts = torch.Tensor(np.random.randint(2,size=(1024,1)))

    t0 = time.time()
    for ii in range(10000):

        
        y_pred = model.forward(x)
        loss = ce_loss(y_tgts, y_pred)

        loss.backward()
        with torch.no_grad():
            for params in model.w:
                params -= 1e-2 * params.grad

        # manual zero_grad
        for jj in range(len(model.w)):
            model.w[jj].grad *= 0.0 

        if ii % 100 == 0:
            print(loss)

    t1 = time.time()
    print("pytorch time elapsed: {:.2f}".format(t1-t0))
