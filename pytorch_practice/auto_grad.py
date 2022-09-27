'''
implement logistic regression manually first 
and finally implement pytorch model (replace manual)
'''
import torch 

# f = w * x 
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)   # assume w = 2 => y = 2*x

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)                                     # init weight
''' forward pass'''
# model prediction 
def forward(x):
    return w * x

# loss (mse)
def loss(y, y_hat):
    return ((y_hat-y)**2).mean()

print(f"Prediction before training: f(5) {forward(5):.3f}")

# start training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss 
    l = loss(Y, y_pred)

    # gradient 
    # dw = gradient(y_pred, X, Y)
    l.backward()

    # update weight (GD: -gradient*learning_rate)
    # this operation is not part of grad tracking graph 
    with torch.no_grad():
        w -= learning_rate * w.grad

    # avoid accumulation 
    w.grad.zero_()

    # print every 10th step 
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.3f}')
print(f"Prediction after training: f(5) {forward(5):.3f}")
