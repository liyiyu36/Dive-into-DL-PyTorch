import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

torch.manual_seed(seed=3)
np.random.seed(seed=3)
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
# features 的每一行是长度为 2 的向量
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
# labels 的每一行是一个长度为 1 的向量（标量）
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 噪声项 epsilon 服从均值为 0 标准差为 0.01 的正态分布
# 噪声代表了数据集中无意义的干扰
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
print(features[0], labels[0])  # tensor([-0.0766,  0.3599]) tensor(2.8410)


# 生成第二个特征 features[:, 1] 和标签 labels 的散点图 可以更直观地观察两者间的线性关系
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个 batch_size
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    # tensor([[-0.2069, -0.2415],
    #         [-0.7246, -0.9330],
    #         [ 0.4915,  0.4856],
    #         [ 2.4667, -1.0854],
    #         [ 0.3715, -1.2071],
    #         [ 0.9260,  0.5741],
    #         [-0.1150, -0.9088],
    #         [-0.2923, -0.0641],
    #         [-0.3985,  0.2452],
    #         [ 0.7545,  0.4476]])
    # tensor([ 4.6167,  5.9106,  3.5423, 12.8254,  9.0492,  4.0975,  7.0600,  3.8236,
    #          2.5680,  4.1907])
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义模型
def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


# 训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    # epoch 1, loss 0.042475
    # epoch 2, loss 0.000176
    # epoch 3, loss 0.000051

# 训练完成后 我们可以比较学到的参数和用来生成训练集的真实参数
print(true_w, '\n', w)
print(true_b, '\n', b)
# [2, -3.4]
#  tensor([[ 1.9996],
#         [-3.3987]], requires_grad=True)
# 4.2
#  tensor([4.2001], requires_grad=True)
