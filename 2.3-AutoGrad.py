import torch

# 创建一个 tensor 并设置 requires_grad=True
x = torch.ones(2, 2, requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
print(x.grad_fn)    # x 是直接创建的 所以没有 grad_fn
# None

# 加法操作
y = x + 2
print(y)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
print(y.grad_fn)    # y 是通过一个加法创建的 所以有一个名为 <AddBackward0> 的 grad_fn
# <AddBackward0 object at 0x000001B6A033F828>

# x 这种直接创建的称为 叶子节点 叶子节点对应的 grad_fn 为 None
print(x.is_leaf, y.is_leaf)
# True False

# 稍复杂一点的运算
z = y * y * 3
out = z.mean()
print(z, out)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)
# tensor(27., grad_fn=<MeanBackward0>)


# 通过 .requires_grad_() 来用 in-place 的方式改变 requires_grad 属性
a = torch.randn(2, 2)   # 缺失情况下默认 requires_grad = False
a = (a * 3) / (a - 1)
print(a.requires_grad)
# False
a.requires_grad_(True)
print(a.requires_grad)
# True
b = (a * a).sum()
print(b.grad_fn)
# <SumBackward0 object at 0x0000021FA2A1FD68>
c = b.mean()
print(c.grad_fn)
# <MeanBackward0 object at 0x000001C47738FD68>


# 梯度
# out 是一个标量 调用 backward() 时不需要指定求导变量
out.backward()  # 等价于 out.backward(torch.tensor(1.))
# out 关于 x 的梯度 d(out)/dx
print(x.grad)
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])

# grad 在反向传播过程中是累加的 (accumulated)
# 每一次运行反向传播 梯度都会累加之前的梯度 所以一般在反向传播之前需把梯度清零
out2 = x.sum()
out2.backward()
print(x.grad)
# tensor([[5.5000, 5.5000],
#         [5.5000, 5.5000]])


# example
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
# tensor([[2., 4.],
#         [6., 8.]], grad_fn=<ViewBackward>)

# 现在 z 不是一个标量
# 所以在调用 backward 时需要传入一个和 z 同形的权重向量进行加权求和得到一个标量
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)   # x.grad 是和 x 同形的张量
# tensor([2.0000, 0.2000, 0.0200, 0.0020])


# 中断梯度追踪
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)          # True
print(y1, y1.requires_grad)
# tensor(1., grad_fn=<PowBackward0>) True
print(y2, y2.requires_grad)
# tensor(1.) False
# y2.requires_grad=False 所以不能调用 y2.backward() 会报错
print(y3, y3.requires_grad)
# tensor(2., grad_fn=<AddBackward0>) True

y3.backward()
print(x.grad)
# tensor(2.)

# y2.backward()
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


# 如果我们想要修改 tensor 的数值 但是又不希望被 autograd 记录(即不会影响反向传播)
# 那么可以对 tensor.data 进行操作
x = torch.ones(1, requires_grad=True)
print(x.data)                   # 还是一个tensor
# tensor([1.])
print(x.data.requires_grad)     # 但是已经是独立于计算图之外
# False

y = 2 * x
x.data *= 100                   # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x)
# tensor([100.], requires_grad=True)
print(x.grad)
# tensor([2.])
