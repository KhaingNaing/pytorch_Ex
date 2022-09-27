''' Convolutional Neural network 

    similar to Neural Network, they are made up of neurons 
    that have learnable weights and biases. Main diff is 
    CNN mainly work on image data and apply convolutional 
    filter (typical confident architecture) 
    (conv layer -> activation layer -> pooling layer and finally Fully connected layer at the end)
'''
import torch  
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# hyper parameters
n_epoches = 4
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]
# we transfrom them to Tensors of normalized range [-1, 1]
transform = transforms.Compose()

# implement CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d()
        self.pool = nn.MaxPool2d()
        self.conv2 = nn.Conv2d()
        self.fcl = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten 
        x = x.view(-1, )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
