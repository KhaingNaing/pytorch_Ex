'''
    Softmax Layer or Cross Entropy 
'''
from asyncio import create_subprocess_exec
from audioop import cross
import numpy as np 
import torch 
import torch.nn as nn

# in numpy 
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy: ', outputs)

# in torch 
x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x, dim=0)           # must specify dimension
print('softmax torch: ', outputs)

# a lot of time softmax is combined with cross-entropy loss
# better the prediction => lower the entropy loss


''' Cross Entropy '''
# in numpy 
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y is one hot encoded 
# class1 = [1 0 0]
# class2 = [0 1 0]
# class3 = [0 0 1]
Y = np.array([1,0,0])

# y_pred has probabilities
y_pred_good = np.array([0.7,0.2,0.1])   # P(class1) high
y_pred_bad = np.array([0.1,10.3,0.6])   # P(class1) low
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
print(f'Loss1 numpy: {l1:.2f} (as loss(entropy) small => good prediction)')
print(f'Loss2 numpy: {l2:.2f} (as loss(entropy) large => bad prediction)')

'''
    Careful!!!!
    - nn.CrossEntropyLoss applies (nn.LogSoftmax + nn.NLLLoss(negative log likelihood loss))
    must not 
    - no softmax layer implementation by us 
    - Y not One-Hot (must have class labels)
    - y_pred no softmax (has raw scores (logits))

'''
# in pytorch 
loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])   # not one-hot (correct class label only)

# nsamples x nclasses = 1x3
# here we have 1 sample and 3 possible classes
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])       # no softmax! (raw values)
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, pred1 = torch.max(y_pred_good, 1)
_, pred2 = torch.max(y_pred_bad, 1)

print(pred1)
print(pred2)