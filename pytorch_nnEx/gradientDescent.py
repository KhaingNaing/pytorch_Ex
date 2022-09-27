''' Gradient Descent with numpy '''

import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt

# linear function y = 1 + 5*x with some Gaussian(normal) noise added 

''' Generate Data '''
x = np.random.rand(100, 1)      # generate random numbers from the standard uniform distribution from 0 to 1
y = 1 + 5 * x + 0.1*np.random.randn(100, 1)

''' 
X = [[1, x1],
     [1, x2],
     [1, x3],
     ...,
     [1, xn]]
y = [[y1],
     [y2],
     ...
     [yn]]
w = [[w0],
     [w1]]
'''
X = np.concatenate((np.ones((100,1)),x), axis=1)
print(X.shape)
w = np.zeros((2, 1))
print(w.shape)

def predict(X, w):
    return X@w

def SSEloss(y, y_pred):
    return 1/2*(y_pred-y)**2

def gradient(X, y, y_pred):
    return -1*X.T@(y-y_pred)

learning_rate = 0.001
num_epoches = 1000

for epoch in range(num_epoches):
    y_predicted = predict(X, w)

    loss = SSEloss(y, y_predicted)

    dw = gradient(X, y, y_predicted)

    w -= learning_rate*dw
print(w)

''' Visualize the data '''
plt.plot(x, y, 'ro')
plt.plot(X, y_predicted, 'b--')
plt.show()

# [[1.03462956][4.95343438]]