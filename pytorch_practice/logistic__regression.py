''' pytorch logistic regression 
    1) Design Model (input_size, output_size, forward pass)
    2) Construct loss and optimizer
    3) Training loop
        - forward pass: compute prediction and loss
        - backward pass: gradients
        - update weights
        iterate until done
'''
from distutils.log import Log
import torch  
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# prepare data (construct a binary classification problem)
breastCancer_db = datasets.load_breast_cancer()
X, y = breastCancer_db.data, breastCancer_db.target

n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# scale features (scale to 0 mean (recommend to do for logistic regression)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# convert to torch.tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)     # reshape y from row to column vector 
y_test = y_test.view(y_test.shape[0], 1)

# 1) Design Model 
''' model: f = w*x + bias, sigmoid funciton is applied at the end (logistic regression)
'''
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        # super constructor 
        super(LogisticRegression, self).__init__()

        # define layer 
        self.linear = nn.Linear(n_input_features, 1)        # input_dim = n_input_features; output_dim = 1 (1 class label at the end)

    def forward(self, x):
        ''' 
        for logistic regression:
            first apply linear layer 
            then sigmoid function at the end
        '''
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)      # 30 input features and 1 output feature

# 2) Construct loss and optimizer 
learning_rate = 0.01
criterion = nn.BCELoss()                         # Binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop 
n_epoches = 100
for epoch in range(n_epoches):
    # forward pass (prediction and loss)
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass (gradients)
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()       # empty grad 

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# evaluation (must not be tracked in our computational history)
with torch.no_grad():
    y_predicted = model(X_test)
    # sigmoid function will return a value between 0 and 1
    # therefore round y_predicted into two classes 0 or 1
    y_predicted_class = y_predicted.round()
    acc = y_predicted_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')


# play around with epoches, learning_rate or diff optimizer for better accuracy 