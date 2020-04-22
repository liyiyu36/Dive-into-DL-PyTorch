import torch.nn as nn
import torch
import numpy as np
# 读取数据
import torch.utils.data as Data

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break


# tensor([[-0.3023, -1.8630],
#         [ 0.8327,  0.3132],
#         [-1.0049, -1.0176],
#         [ 0.7469,  0.5573],
#         [ 0.0060,  0.5198],
#         [ 0.4674, -0.6867],
#         [-1.6883,  0.1735],
#         [ 0.7981,  0.1989],
#         [ 0.9273, -0.9457],
#         [ 1.0236, -0.4228]])
# tensor([9.9188, 4.8115, 5.6461, 3.8097, 2.4489, 7.4663, 0.2285, 5.1155, 9.2720,
#         7.6938])

# 定义模型
# nn 的核心数据结构是 Module 它是一个抽象概念 既可以表示神经网络中的某个层（layer） 也可以表示一个包含很多层的神经网络
# 在实际使用中 最常见的做法是继承 nn.Module 撰写自己的网络/层
# 一个 nn.Module 实例应该包含一些层以及返回输出的前向传播（forward）方法
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)
# LinearNet(
#   (linear): Linear(in_features=2, out_features=1, bias=True)
# )


# 还可以用 nn.Sequential 来更方便地搭建网络
# Sequential 是一个有序的容器 网络层将按照在传入 Sequential 的顺序依次被添加到计算图中
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
)
# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ...
# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
    ('linear', nn.Linear(num_inputs, 1))
    # ......
]))

print(net)
# Sequential(
#   (linear): Linear(in_features=2, out_features=1, bias=True)
# )
print(net[0])
# Linear(in_features=2, out_features=1, bias=True)

# 可以通过 net.parameters() 来查看模型所有的可学习参数
# net.parameters() 将传回一个生成器
for param in net.parameters():
    print(param)
# Parameter containing:
# tensor([[ 0.6315, -0.6309]], requires_grad=True)
# Parameter containing:
# tensor([-0.3480], requires_grad=True)

# 注：torch.nn 仅支持输入一个 batch 的样本不支持单个样本输入 如果只有单个样本 可使用 input.unsqueeze(0) 来添加一维


# 初始化模型参数
# 通过 init.normal_ 将权重参数每个元素初始化为随机采样于均值为 0 标准差为 0.01 的正态分布
# 偏差会初始化为 0
from torch.nn import init
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)      # 也可以直接修改 bias 的 data : net[0].bias.data.fill_(0)

# 注: 如果 net 自定义 则会报错 net[0].weight 应改为 net.linear.weight bias 亦然
# 因为 net[0] 这样根据下标访问子模块的写法只有当 net 是个 ModuleList 或者 Sequential 实例时才可以


# 定义损失函数
# 均方误差损失
loss = nn.MSELoss()


# 定义优化算法
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
# SGD (
# Parameter Group 0
#     dampening: 0
#     lr: 0.03
#     momentum: 0
#     nesterov: False
#     weight_decay: 0
# )

# 还可以为不同子网络设置不同的学习率 这在 finetune 时经常用到
# optimizer = optim.SGD([
#     # 如果对某个参数不指定学习率 就使用最外层的默认学习率
#     {'params': net.subnet1.parameters()},       # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr':0.01}
# ], lr=0.03)
# AttributeError: 'Sequential' object has no attribute 'subnet1'


# 不将学习率设置成固定的常数 可以调整学习率
# 方法一: 修改 optimizer.param_groups 中对应的学习率
# 方法二: 新建优化器
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1    # 学习率为之前的 0.1 倍


# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()       # 梯度清零 等价于 net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
# epoch 1, loss: 6.773077
# epoch 2, loss: 3.318739
# epoch 3, loss: 0.708743

# 比较学到的模型参数和真实的模型参数
# 从 net 获得需要的层 并访问其权重 weight 和偏差 bias
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
# [2, -3.4] Parameter containing:
# tensor([[ 1.7138, -2.7873]], requires_grad=True)
# 4.2 Parameter containing:
# tensor([3.5294], requires_grad=True)








