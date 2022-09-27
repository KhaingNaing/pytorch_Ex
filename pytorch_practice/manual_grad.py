'''
implement logistic regression manually first 
and finally implement pytorch model (replace manual)
'''
import numpy as np 

# f = w * x 
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)   # assume w = 2 => y = 2*x

w = 0.0                                     # init weight
''' forward pass'''
# model prediction 
def forward(x):
    return w * x

# loss (mse)
def loss(y, y_hat):
    return ((y_hat-y)**2).mean()

# grad (note y_hat = w*x)
#     MSE = 1/N * (w*x-y)**2
#     dloss/dw = 1/N * 2 * x * (w*x-y)
def gradient(y_hat, x, y):
    return np.dot(2*x, y_hat-y).mean()

print(f"Prediction before training: f(5) {forward(5):.3f}")

# start training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss 
    l = loss(Y, y_pred)

    # gradient 
    dw = gradient(y_pred, X, Y)

    # update weight (GD: -gradient*learning_rate)
    w -= learning_rate * dw

    # print every step 
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.3f}')
print(f"Prediction after training: f(5) {forward(5):.3f}")
