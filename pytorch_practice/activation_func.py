'''
step func 
    - not use in practice for nn

sigmoid 
    - typically in the last layer of a binary classfication nn

Hyperbolic Tangent 
    - Hidden Layers

ReLU (rule of thumb!)
    - most popular (use a ReLU for hidden layers)

Leaky ReLU
    - improved version (try to solve the vanishing gradient problem)

Softmax
    - good in last layer in multi class classification problems 
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
#F.leaky_relu


### Binary problem 

# option 1 (create nn Modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # for hidden layer (relu)
        self.relu = nn.ReLU()                       
        self.linear2 = nn.Linear(hidden_size, 1)
        # for last layer (sigmoid)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out 

# option 2 (use activation funcs directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)                   
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # apply ReLU on first layer
        out = nn.ReLU(self.linear1(x))
        # apply Sigmoid on last layer 
        out = nn.Sigmoid(self.linear2(out))
        return out 