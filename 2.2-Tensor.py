import torch
import numpy as np

# empty() 保留内存中的原始内容 不做初始化
x = torch.empty(5, 3)
print(x)
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]])

# 随机数据
x = torch.rand(5, 3)
print(x)
# tensor([[0.0098, 0.8724, 0.1224],
#         [0.3223, 0.2987, 0.5762],
#         [0.6201, 0.4258, 0.1718],
#         [0.1944, 0.3227, 0.1466],
#         [0.0085, 0.8536, 0.1422]])

# 全零张量 数据类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])

# 直接根据数据创建
x = torch.tensor([5.5, 3])
print(x)
# tensor([5.5000, 3.0000])

# 通过现有的 Tensor 来创建 默认重用输入 Tensor 的一些属性 如数据类型 除非自定义数据类型
x = x.new_ones(5, 3, dtype=torch.float64)   # x 具有和原来的 x 默认相同的 torch.dtype 和 torch.device
print(x)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)

# 指定新的数据类型
x = torch.rand_like(x, dtype=torch.float)
print(x)
# tensor([[0.6405, 0.6797, 0.1563],
#         [0.4837, 0.0867, 0.5221],
#         [0.8649, 0.4590, 0.0491],
#         [0.1064, 0.9314, 0.9415],
#         [0.5260, 0.2630, 0.3745]])

# 返回的 torch.size 其实是一个 tuple 支持所有的 tuple 操作
print(x.shape)
# torch.Size([5, 3])
print(x.size())
# torch.Size([5, 3])




# 加法形式一
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
# tensor([[1.2025, 1.3015, 1.6289],
#         [0.7458, 1.3650, 1.8830],
#         [1.0193, 1.0927, 1.4043],
#         [1.8015, 0.5634, 1.0920],
#         [0.9248, 0.9747, 1.2452]])

# 加法形式二 指定输出
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# tensor([[1.2025, 1.3015, 1.6289],
#         [0.7458, 1.3650, 1.8830],
#         [1.0193, 1.0927, 1.4043],
#         [1.8015, 0.5634, 1.0920],
#         [0.9248, 0.9747, 1.2452]])

# 加法形式三 inplace
# PyTorch 操作 inplace 版本都有后缀 _ 例如 x.copy_(y) x.t_()
y.add_(x)
print(y)
# tensor([[1.2025, 1.3015, 1.6289],
#         [0.7458, 1.3650, 1.8830],
#         [1.0193, 1.0927, 1.4043],
#         [1.8015, 0.5634, 1.0920],
#         [0.9248, 0.9747, 1.2452]])


# 索引
# 索引出来的结果与原数据共享内存 也即修改一个 另一个也会跟着修改
y = x[0, :]
print(x)
# tensor([[0.2591, 0.1555, 0.7945],
#         [0.4077, 0.8821, 0.1004],
#         [0.4661, 0.0567, 0.0105],
#         [0.1385, 0.6767, 0.2284],
#         [0.9144, 0.2573, 0.8170]])
print(y)
# tensor([0.2591, 0.1555, 0.7945])
y += 1
print(y)
# tensor([1.2591, 1.1555, 1.7945])
print(x[0, :])  # 源 tensor 也被改了
# tensor([1.2591, 1.1555, 1.7945])


# 用 view() 来改变形状
y = x.view(15)
z = x.view(-1, 5)   # -1 所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
# torch.Size([5, 3]) torch.Size([15]) torch.Size([3, 5])

# 新 tensor 与原 tensor 可能 size 不同 但是是共享 data 的  更改其中一个 另一个也跟着改变
x += 1
print(x)
# tensor([[2.5910, 2.5196, 2.1485],
#         [1.1054, 1.4511, 1.0598],
#         [1.5963, 1.9556, 1.7510],
#         [1.2055, 1.7454, 1.9487],
#         [1.1516, 1.2624, 1.9470]])
print(y)
# tensor([2.5910, 2.5196, 2.1485, 1.1054, 1.4511, 1.0598, 1.5963, 1.9556, 1.7510,
#         1.2055, 1.7454, 1.9487, 1.1516, 1.2624, 1.9470])


# reshape() 可以改变 tensor 的形状 但是不能保证返回的是其拷贝
# 想要返回一个真正的副本 不共享 data 可以使用 clone 创造一个副本 再 view
# 使用 clone 还有个好处 会被记录进计算图中 梯度回传到副本时 也会回传到源 tensor
x_cp = x.clone().view(15)
x -= 1
print(x)
# tensor([[1.7478, 1.0653, 1.0936],
#         [0.2610, 0.4783, 0.0852],
#         [0.2353, 0.3836, 0.7979],
#         [0.5439, 0.4776, 0.3335],
#         [0.8929, 0.4631, 0.0703]])
print(x_cp)
# tensor([2.7478, 2.0653, 2.0936, 1.2610, 1.4783, 1.0852, 1.2353, 1.3836, 1.7979,
#         1.5439, 1.4776, 1.3335, 1.8929, 1.4631, 1.0703])


# item() 可以将一个标量 tensor 转换成一个 Python number
x = torch.randn(1)
print(x)
# tensor([-2.6127])
print(x.item())
# -2.6127405166625977


# 当两个 shape 不同的 tensor 做 element-wise 的运算时 可能会触发广播机制 broadcasting
# 先适当复制元素使两个 tensor shape 相同 再做 element-wise 的运算
x = torch.arange(1, 3).view(1, 2)
print(x)
# tensor([1, 2])
y = torch.arange(1, 4).view(3, 1)
print(y)
# tensor([[1],
#         [2],
#         [3]])
print(x + y)
# tensor([[2, 3],
#         [3, 4],
#         [4, 5]])


# python 自带了 id 函数 如果两个实例的 id 相同 则对应相同的内存地址
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
# 1516543943376
y = y + x
id_after = id(y)
print(id_after)
# 1493355206480
print(id(y) == id_before)
# False

# 如果想指定结果到原来 y 的内存中 可以用前面的索引来进行替换操作
# 把 x + y 通过[:] 写进 y 对应的内存中
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
print(id_before)
# 2267179028128
y[:] = y + x
print(id(y))
# 2267179028128
print(id(y) == id_before)
# True

# 运算符全名函数中的 out 参数 或者自加运算符 += 或者 add_() 函数也能达到上述效果
# torch.add(x, y, out=y)    y += x  y.add_(x)
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y)  # y += x    y.add_(x)
print(id(y) == id_before)
# True
# view() 虽然和源 tensor 共享 data 但是并不是一个 tensor
# tensor 相同除了 data 相同外 还需要其他的属性相同 如 id(内存地址)


# Tensor 和 Numpy 相互转换
# 我们很容易用 numpy() 和 from_numpy() 将 Tensor 和 NumPy 中的数组相互转换
# 这两个函数所产生的的 Tensor 和 NumPy 中的数组共享相同的内存（所以他们之间的转换很快） 改变其中一个时另一个也会改变！！！

# tensor -> numpy
a = torch.ones(5)
b = a.numpy()
print(a, b)
# tensor([1., 1., 1., 1., 1.]) [1. 1. 1. 1. 1.]
a += 1
print(a, b)
# tensor([2., 2., 2., 2., 2.]) [2. 2. 2. 2. 2.]
b += 1
print(a, b)
# tensor([3., 3., 3., 3., 3.]) [3. 3. 3. 3. 3.]

# numpy -> tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
# [1. 1. 1. 1. 1.] tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
a += 1
print(a, b)
# [2. 2. 2. 2. 2.] tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
b += 1
print(a, b)
# [3. 3. 3. 3. 3.] tensor([3., 3., 3., 3., 3.], dtype=torch.float64)
# 所有在 CPU 上的 Tensor （除了CharTensor）都支持与 NumPy 数组相互转换

# 直接用 torch.tensor() 将 NumPy 数组转换成 Tensor
# 该方法总是会进行数据拷贝 返回的 Tensor 和原来的数据不再共享内存
c = torch.tensor(a)
a += 1
print(a, c)
# [4. 4. 4. 4. 4.] tensor([3., 3., 3., 3., 3.], dtype=torch.float64)


# tensor on GPU
# 用方法 to() 可以将 Tensor 在 CPU 和 GPU （需要硬件支持）之间相互移动
if torch.cuda.is_available():
    device = torch.device("cuda")           # GPU
    y = torch.ones_like(x, device=device)   # x = tensor([1, 2])
    # y = tensor([1, 1], device='cuda:0')
    x = x.to(device)                        # 等价于 x = x.to("cuda")
    z = x + y
    print(z)
    # tensor([2, 3], device='cuda:0')
    print(z.to("cpu", dtype=torch.double))  # to() 同时还可以更改数据类型
    # tensor([2., 3.], dtype=torch.float64)



