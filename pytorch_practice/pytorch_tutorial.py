# In pytorch everything is based on tensor operation
import torch 
import numpy as np
'''  Ways to create tensor 
tinput: dtype = float by default
input: requires_grad = True (False by default)

torch.empty(size)
    returns a tensor filled with uninitialized data
    eg. torch.empty(2,3) => 2d

torch.rand(size)
    returns a tensor filled with random values drawn from unifrom distribution [0, 1)
torch.randn(size)
    returns a tensor filled with random values drawn from Guassian distribution (mean = 0, variance = 1)

torch.zeros(size)
torch.ones(size, dtype=torch.double)

torch.tensor([python list of data])
'''
t = torch.ones(2,2, dtype=torch.double)
print("torch.ones() ", t.size(), t.dtype)

t = torch.tensor([2.5, 0.1])
print(t, t.size())

''' 
basic operations on torch 
x  = torch.rand; y = torch.rand 

    addition:
        + => elementwise addition 
        torch.add(x, y) => same as '+'
        y.add_(x) => in place addition 
    substraction:
        x - y
        torch.sub(x, y)
        y.sub_()
    mul(), div() etc

    Note: in pytorch => every funciton that has a trailing underscore will do in place operation 
        in place operation => will modify the variable (y) that it is applied on 

slicing operations 
   eg. x[:, 1] => all rows at col 1

reshaping
    x.view(new_shape) 
    eg. x.view(-1, 8) => pytorch will determine right size for row (with col=8)
'''

''' 
if using cpu both numpy and torch objs has the same memory => changing one will change the other 

converting numpy to torch 
    b.torch.from_numpy(..numpy..)

converting torch to numpy  
    a.nump(..torch..) 

'''

# gradient in pytorch 
x = torch.randn(3, requires_grad=True)

y = x + 2           # pytorch will create the computation graph for us 
print(y)
z = y*y*2
print(z)

v = torch.tensor([0.1,1.0,0.001])
z.backward(v)       # dz/dx calculate gradient (note z is Jacobian product(chain rule)) 
print(x.grad)

''' prevent pytorch from tracking the gradient 
opt1:
    x.requires_grad_(False)
opt2:
    x.detach()
opt3: 
    with torch.no_grad():
        # do operations
'''

''' grad for the tensor will be accumulated in the .grad attribute (the value will be sum up)
==> careful  (useful while using optimizer from pytorch => must emputy grad)
'''
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    print("model_output = ",model_output)

    model_output.backward()
    print("grad = ", weights.grad)

    # before next iteration => empty the grad to avoid accumulation 
    weights.grad.zero_()


''' Backpropagation in pytorch '''
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass => compute the loss func
y_hat = w * x
loss = (y_hat - y)**2

print("loss = ", loss)

# backward pass
loss.backward()
print(w.grad)

# then upate weight and do next forward & backward 



