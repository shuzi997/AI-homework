# -*- coding: utf-8 -*-
"""
零基础版：用 numpy 手写一个最简单的 MLP（多层感知机）
结构完全按照课本第 16 节的图：输入层 2 个神经元 -> 隐藏层 5 个 -> 输出层 1 个
任务：二分类，最后画出训练过程中 Loss 的下降折线图
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. 随便编一组简单的分类数据 ----------
np.random.seed(0)                       # 固定随机种子，每次运行结果一样

# 第 0 类：50 个点，聚在 (0, 0) 附近
X0 = np.random.randn(50, 2) + [0, 0]
# 第 1 类：50 个点，聚在 (3, 3) 附近
X1 = np.random.randn(50, 2) + [3, 3]

X = np.vstack([X0, X1])                 # 数据 X，形状 (100, 2)：100 个点，每个点 2 个特征
y = np.vstack([np.zeros((50, 1)),       # 前 50 个点标签是 0
               np.ones((50, 1))])       # 后 50 个点标签是 1，形状 (100, 1)

# ---------- 2. 定义网络参数（对照书上的三层结构图）----------
def sigmoid(x):                         # 激活函数
    return 1 / (1 + np.exp(-x))

W1 = np.random.randn(2, 5) * 0.5        # 输入层 -> 隐藏层 的权重
b1 = np.zeros(5)                        # 隐藏层 的偏置
W2 = np.random.randn(5, 1) * 0.5        # 隐藏层 -> 输出层 的权重
b2 = np.zeros(1)                        # 输出层 的偏置

lr = 0.5                                # 学习率（每次参数更新的步子大小）
losses = []                             # 用一个列表记录每一轮的 Loss

# ---------- 3. 训练：反复做 "前向传播 -> 算 Loss -> 反向传播 -> 更新参数" ----------
for epoch in range(1000):

    # --- 前向传播：数据从输入层一层层算到输出层 ---
    z1 = X @ W1 + b1                    # 隐藏层的加权和
    h = sigmoid(z1)                     # 隐藏层输出，形状 (100, 5)
    z2 = h @ W2 + b2                    # 输出层的加权和
    out = sigmoid(z2)                   # 网络的预测值，形状 (100, 1)

    # --- 计算 Loss（均方误差：预测值和真实标签差多少）---
    loss = np.mean((out - y) ** 2)
    losses.append(loss)                 # 记下来，最后画图用

    # --- 反向传播：算出每个参数该往哪个方向调（梯度）---
    d_out = (out - y) * out * (1 - out)         # 输出层的误差信号
    d_W2 = h.T @ d_out                          # W2 的梯度
    d_b2 = d_out.sum(axis=0)                    # b2 的梯度
    d_h = d_out @ W2.T * h * (1 - h)            # 误差传回隐藏层
    d_W1 = X.T @ d_h                            # W1 的梯度
    d_b1 = d_h.sum(axis=0)                      # b1 的梯度

    # --- 沿着梯度的反方向更新参数（梯度下降）---
    W1 -= lr * d_W1
    b1 -= lr * d_b1
    W2 -= lr * d_W2
    b2 -= lr * d_b2

# ---------- 4. 看看训练得怎么样 ----------
pred = (out > 0.5).astype(int)          # 预测值大于 0.5 就算第 1 类
print("最后的 Loss:", round(losses[-1], 4))
print("训练集准确率:", (pred == y).mean() * 100, "%")

# ---------- 5. 画 Loss 下降折线图 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']    # 让图里能显示中文
plt.rcParams['axes.unicode_minus'] = False

plt.plot(losses, color='royalblue')
plt.xlabel("训练轮数")
plt.ylabel("Loss（损失值）")
plt.title("MLP 训练过程中 Loss 的下降曲线")
plt.grid(True, alpha=0.4)
plt.show()
