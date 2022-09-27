'''
Transforms can be applied to PIL images, tensors, ndarrays or custom data 
during creation of the Dataset

complete list of bulit-in transforms:
https://pytorch.org/docs/stable/torchvision/transforms.html

torchvision.transforms.ReScale(256)
torchvision.transforms.ToTensor()
'''
import torch 
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math 
import csv

# buildin dataset can use transform argument to apply tensor transform 
#dataset = torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor())

# custom dataset (inherit Dataset) extended to support transform
class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data Loading (file is separated by "," and skip the header row)
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]        # don't want first col (col 0)
        self.y = xy[:, [0]]      # just the first col (col 0) => [:, [0]] = (n_samples, 1)
        self.n_samples = xy.shape[0]

        # optional transform (if it is available)
        self.transform = transform

    def __getitem__(self, index):
        # allow for indexing (return tuple)
        sample = self.x[index], self.y[index]

        # optional transform (if it is available)
        if self.transform:
            sample = self.transform(sample)
        return sample 

    def __len__(self):
        # len(dataset)
        return self.n_samples

# custom transform
class ToTensor():
    # callable object 
    def __call__(self, sample):
        inputs, targets = sample 
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample 
        inputs  *= self.factor 
        return inputs, targets

dataset = WineDataset(transform=ToTensor())
#dataset = WineDataset(transform=None)

first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

# composed transform
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)


