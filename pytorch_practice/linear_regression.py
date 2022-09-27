''' Typical pytorch model 
    1) Design model (input_size, output_size, forward pass)
    2) Construct loss and optimizer function 
    3) Training loop
        - forward pass: compute prediction and loss
        - backward pass: gradients
        - update weights
        iterate until done 
'''
from pickletools import optimize
import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare Data (generate a regression dataset with 100 samples and 1 feature)
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
 
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))
print(Y.size())
Y = Y.view(Y.shape[0], 1)                   # reshape column vector into 2d 
print(Y.size())

n_samples, n_features = X.shape

# 1) Design Model 
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)  # buildin model

# 2) Define loss and optimizer 
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop 
n_epoches = 100

for epoch in range(n_epoches):
    # forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    # backward pass
    loss.backward()

    # weights update
    optimizer.step()

    optimizer.zero_grad()       # empty grad before next grad

    if (epoch+1) % 10 == 0:     # print every 10th steps
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()   # Avoid tracking this operation in the computational graph 
plt.plot(X_numpy, y_numpy, 'ro')        # first plot X and y data
plt.plot(X_numpy, predicted, 'b')       # plot generated or approximated function
plt.show()