import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 生成随机数据，模拟二维特征空间的数据点
np.random.seed(0)
data = np.random.randn(100, 2)
# Step 1: 数据中心化（减去均值）
mean = np.mean(data, axis=0)
centered_data = data - mean
# Step 2: 计算协方差矩阵
cov_matrix = np.cov(centered_data, rowvar=False)
# Step 3: 求协方差矩阵的特征值和特征向量
values, vectors = np.linalg.eigh(cov_matrix)
# Step 4: 按特征值大小降序排序
sorted_indices = np.argsort(values)[::-1]
sorted_values = values[sorted_indices]
sorted_vectors = vectors[:, sorted_indices]
# Step 5: 投影数据到新基空间（选择前两个主成分）
k = 1  # 降维到1维
selected_vectors = sorted_vectors[:, :k]
trans_data = centered_data.dot(selected_vectors)
# 输出结果
print("投影后的数据：\n", trans_data)
# 原始数据
plt.scatter(data[:, 0], data[:, 1], label='Original Data', alpha=0.5)
# 投影后的数据（1D表示）
plt.scatter(trans_data[:, 0], np.zeros_like(trans_data[:, 0]), color='red', label='K-L Transform Data')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.title("K-L Transform ")
plt.show()
