import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('Cameraman.tif', cv2.IMREAD_GRAYSCALE)
# Step 1: 数据中心化（减去均值）
mean = np.mean(image, axis=0)
centered_image = image - mean
# Step 2: 计算协方差矩阵
cov_matrix = np.cov(centered_image, rowvar=False)
# Step 3: 求协方差矩阵的特征值和特征向量
values, vectors = np.linalg.eigh(cov_matrix)
# Step 4: 按特征值大小降序排序
sorted_indices = np.argsort(values)[::-1]
sorted_values = values[sorted_indices]
sorted_vectors = vectors[:, sorted_indices]
# 定义不同的特征值数目来进行降维
k_values = [8, 16, 32, 64]
re_images = []
for k in k_values:
    # Step 5: 选择前 k 个特征向量
    selected_vectors = sorted_vectors[:, :k]
    # 投影图像到 k 维空间
    trans_data = centered_image.dot(selected_vectors)
    # 将图像重建回原空间
    re_image = np.dot(trans_data, selected_vectors.T) + mean
    re_images.append(re_image)

# 显示原图像和重建图像
plt.figure(figsize=(10, 8))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

for i, (k, img) in enumerate(zip(k_values, re_images), start=2):
    plt.subplot(2, 3, i)
    plt.imshow(img, cmap='gray')
    plt.title(f"k = {k}")
    plt.axis('off')
plt.suptitle("K-L Transform Reconstructed Images")
plt.show()
