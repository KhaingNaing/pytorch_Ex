'''
General training pipeline 
    1) Design model (input size, output size, forward pass(layers))
    2) construct loss and optimizer 
    3) training loop 
        - forward pass: compute prediction 
        - backward pass: gradients
        - update weights
        iterate until done 
'''
import torch 
import torch.nn as nn

# f = w * x 
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)   # assume w = 2 => y = 2*x

n_samples, n_features = X.shape
print(n_samples, n_features)

''' pytorch model '''
input_size = n_features
output_size = n_features
#model = nn.Linear(input_size, output_size)
''' now weights = model.parameters()'''

''' custom model '''
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        # super constructor
        super(LinearRegression, self).__init__()

        # define layer 
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_size, output_size)


X_test = torch.tensor([5], dtype=torch.float32)            # dummy test with only one sample 
# for prediction simply call the model(tensor) 
print(f"Prediction before training: f(5) {model(X_test).item():.3f}")

# start training
learning_rate = 0.01
n_iters = 100

''' replace manual loss and weight update(GD) '''
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss 
    l = loss(Y, y_pred)

    # gradient 
    # dw = gradient(y_pred, X, Y)
    l.backward()

    # update weight (GD: -gradient*learning_rate)
    optimizer.step() 

    # avoid accumulation 
    optimizer.zero_grad()

    # print every 10th step 
    if epoch % 10 == 0:
        # unpack for the print
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.3f}')
print(f"Prediction after training: f(5) {model(X_test).item():.3f}")
