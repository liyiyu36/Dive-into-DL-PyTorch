import torch
import torchvision
import numpy as np
import sys
import d2l

# 获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
b = torch.tensor(num_outputs, dtype=torch.float, requires_grad=True)


# 实现 Softmax 运算
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # broadcasting


# 应用
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))


# 定义模型
# 通过 view 函数将每张原始图像改成长度为 num_inputs 的向量
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax)
