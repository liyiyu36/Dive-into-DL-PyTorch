import torch
from time import time

# 矢量计算表达式

a = torch.ones(1000)
b = torch.ones(1000)

# 对两个向量相加的两种方法

# 方法一：将这两个向量按元素逐一做标量加法
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)
# 0.02689647674560547

# 方法二：将这两个向量直接做矢量加法
start = time()
d = a + b
print(time() - start)
# 0.0

# 后者比前者更省时
# 因此 我们应该尽可能采用矢量计算 以提升计算效率


# 广播机制
a = torch.ones(3)
b = 10
print(a + b)
# tensor([11., 11., 11.])