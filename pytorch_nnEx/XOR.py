from heapq import nsmallest
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

learning_rate = 0.1
momentum = 0.0
init = 1.0
epochs = 10000
input  = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
target = torch.Tensor([[0],[1],[1],[0]])

xor_dataset  = TensorDataset(input,target)
train_loader = DataLoader(xor_dataset,batch_size=4)

# binary classification (1 class)
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = self.linear1(x)
        out = torch.tanh(out)
        out = self.linear2(out)
        # Sigmoid (typical binary classification)
        out = torch.sigmoid(out)
        return out 

# hyper parameters
example = iter(train_loader)
samples, labels = example.next()
print(samples.shape, labels.shape)
nsamples, nfeatures = samples.shape
input_size = nfeatures
hidden_size = 2

model = MyModel(input_size, hidden_size)

# initialize weight values
# can we use model.init_params ????
model.linear1.weight.data.normal_(0,init)
model.linear2.weight.data.normal_(0,init)

# Construct optimizer and loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
for epoch in range(1, epochs):

    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward 
        y_pred = model(data)
        loss = F.binary_cross_entropy(y_pred, target)

        # backward
        loss.backward()

        # update weights
        optimizer.step() 

        if epoch % 100 == 0:
            print('ep%3d: loss = %7.4f' % (epoch, loss.item()))
        if loss < 0.01:
            print("Global Mininum")
            exit(0)
print("Local Minimum")
