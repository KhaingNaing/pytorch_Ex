'''
    f = A * x such that f(1) = 1 => find A
'''

import torch 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn

learning_rate = 1.9
momentum = 0.9

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.A = nn.Parameter(torch.zeros((1), requires_grad=True))

    def forward(self, x):
        return self.A * x

input = torch.tensor([[1]])
target = torch.tensor([[1]])

train_dataset = TensorDataset(input, target)
train_loader = DataLoader(train_dataset, batch_size=1)
        
model = MyModel()

# construct loss and optimizer 
def criterion(y_hat, y):
    return 0.5*(((y_hat-y)**2).mean())

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

epochs = 1000
for epoch in range(1,epochs):
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward 
        y_pred = model(data)
        loss = criterion(y_pred, target)

        if type(model.A.grad) == type(None):
            print('Ep%3d: zero_grad(): A.grad=  None  A.data=%7.4f loss=%7.4f' \
                      % (epoch, model.A.data, loss))
        else:
            print('Ep%3d: zero_grad(): A.grad=%7.4f A.data=%7.4f loss=%7.4f' \
                      % (epoch, model.A.grad, model.A.data, loss))


        # backward
        loss.backward()     # compute gradients

        # update weights
        optimizer.step()
        print('            step(): A.grad=%7.4f A.data=%7.4f' \
                    % (model.A.grad, model.A.data))

        if loss < 0.000000001 or np.isnan(loss.data):
            exit(0)    