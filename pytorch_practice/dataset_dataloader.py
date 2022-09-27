''' pytorch dataset and dataloader classes
    - dividing large dataset into smaller batches

terms for batch training:
    - epoch = 1 forward and backward pass of ALL training samples
    - batch_size = # of training samples in one forward & backward pass
    - # of iterations = # of passes, each pass using [batch_size] number of samples
    eg. 100 samples. batch_size=20 --> 100/20 = 5 iterations for 1 epoch 
'''
import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math 
import csv
'''
import pandas as pd

url = 'https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv'
df = pd.read_csv(url,index_col=0,parse_dates=[0])
print(df.head(5))
df.to_csv('wine.csv')
'''
# custom dataset (inherit Dataset)
class WineDataset(Dataset):
    def __init__(self):
        # data Loading (file is separated by "," and skip the header row)
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])        # don't want first col (col 0)
        self.y = torch.from_numpy(xy[:, [0]])       # just the first col (col 0) => [:, [0]] = (n_samples, 1)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # allow for indexing (return tuple)
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataset()
''' following code test the dataset 
first_data = dataset[0]
feature, label = first_data
print(f"first feature: {feature}, label: {label}")
'''

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
''' test dataloader
dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data 
print(features, labels)
'''

# Training loop 
batch_size = 4
num_epoch = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size) # take the ceiling 
print(total_samples, n_iterations)

for epoch in range(num_epoch):

    for i, (inputs, labels) in enumerate(dataloader):
         # forward => backward => update weights
         if (i + 1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epoch}, step {i+1}/{n_iterations}, inputs {inputs.shape}')


