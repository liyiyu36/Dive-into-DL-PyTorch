import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2l

# 获取数据集
# mnist_train 和 mnist_test 都是 torch.utils.data.Dataset 的子类
# 可以用 len() 来获取该数据集的大小
# 训练集中和测试集中的每个类别的图像数分别为 6,000 和 1,000 因为有 10 个类别 所以训练集和测试集的样本数分别为 60,000 和 10,000
mnist_train = torchvision.datasets.FashionMNIST(root='C:/Users/msi-user/Desktop/Dive-into-DL-PyTorch/Datasets'
                                                     '/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='C:/Users/msi-user/Desktop/Dive-into-DL-PyTorch/Datasets'
                                                    '/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
# <class 'torchvision.datasets.mnist.FashionMNIST'>
# 60000 10000

# 通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width
# torch.Size([1, 28, 28]) 9


# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)


start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
# 4.33 sec






