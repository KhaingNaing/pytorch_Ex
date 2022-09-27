# Feed forward Neural Network 
# Multilayer Neural Network for digit classification 

''' MNIST dataset
    1) DataLoader, Transformation
    2) Multilayer Neural Net, activatoin function
    3) Loss and Optimizer
    4) Training loss (batch training)
    5) Model evaluation
    6) GPU support 
'''
from random import shuffle
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784        # later flatten image(28x28) into 1d array 
hidden_size = 100       # you can try diff size
num_classes = 10        # digits (0-9)
n_epoches = 2           # for shorter training (u can set higher)
batch_size = 100        # 100 sample in one batch 
learning_rate = 0.001

# MNIST
# 1) dataloader and transformation 
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(6):
    # subplot with 2 rows with 3 cols = 6 (with index i+1) 
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray') # [i][0] => accessing channel 0 of index i 
plt.show()

# the MNIST digit image has only 1 channel (black or white)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no activation applies at the last layer => for multi class (since using crossEntropy)
        return out 

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop 
n_total_steps = len(train_loader)
for epoch in range(n_epoches):
    for i, (imgs, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        imgs = imgs.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward 
        y_prediction = model(imgs)
        loss = criterion(y_prediction, labels)

        # backward 
        optimizer.zero_grad()       # avoid accumulation
        loss.backward()

        # update parameters 
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1}/{n_epoches}, step: {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# testing and evaluation 
with torch.no_grad():
    n_corrects = 0
    n_samples = 0
    # loop over all the batches in test
    for images, labels in test_loader:
        # reshape images 
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # torch.max() returns (value, index) => we only care about index (class label)
        _, predictions = torch.max(outputs, 1)  # outputs along the dimension 1

        n_samples += labels.shape[0]    # labels.shape[0] of current batch should be 100
        n_corrects += (predictions==labels).sum().item() # for each correct + 1
    
    # total accuracy (percentage)
    acc = 100.0 * n_corrects/n_samples
    print(f'accuracy = {acc}')


        

