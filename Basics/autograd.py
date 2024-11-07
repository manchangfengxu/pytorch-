import torch

# 开启后，自动计算张量变化对应的操作，以便自动求导
x = torch.rand(5, 5, requires_grad=True)
y = torch.rand(5, 5, requires_grad=True)
z = torch.sum(x + y)

print(z.grad_fn)
# output：<SumBackward0 object at 0x0000014BEDF5E340>

z.backward()
print(x.grad, y.grad)