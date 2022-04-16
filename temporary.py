
import torch

data = torch.rand(100, 5)
# This initializes trainable nn.Parameters

lin = torch.nn.Linear(5, 1, bias=True)
# Weight has shape torch.Size([1, 5])
# Bias has shape torch.Size([1])

# Replace the existing weights with tensors of same shape.
# This should just extract the last column of data.
lin.weight.data = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
lin.bias.data = torch.tensor([0.0])

device = ["cpu", "cuda:0"]
for d in device :
    print(d, torch.norm(data[:,4].to(d) - lin.to(d)(data.to(d)).squeeze()))