'''
@Function: 使用Roberts算子进行图像分割实验
@Author: 刘新媛
@Date:2024/12/17
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载并显示图像
image = cv2.imread('Slena.jpg', cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
if image is None:
    raise ValueError("图像加载失败，请检查图像路径是否正确！")

# 显示原始图像
plt.figure(figsize=(8, 6))
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# 2. 使用Roberts算子进行边缘检测
# Roberts算子的水平和垂直模板
rh = np.array([[0, 1], [-1, 0]], dtype=np.float32)  # 水平梯度
rv = np.array([[1, 0], [0, -1]], dtype=np.float32)  # 垂直梯度

# 使用卷积进行滤波
gradient_h = cv2.filter2D(image, -1, rh)  # 水平梯度
gradient_v = cv2.filter2D(image, -1, rv)  # 垂直梯度

# 3. 计算梯度的模（欧几里得距离）
gradient_magnitude = np.sqrt(gradient_h**2 + gradient_v**2)
# 将梯度模转换为uint8类型，并归一化到[0, 255]
gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))

# 显示梯度模（边缘检测结果）
plt.figure(figsize=(8, 6))
plt.imshow(gradient_magnitude, cmap='gray')
plt.title("Edge Detection (Gradient Magnitude)")
plt.axis('off')
plt.show()

# 4. 对梯度模进行二值化处理
# 使用阈值进行二值化
threshold_value = 10
_, binary_edge = cv2.threshold(gradient_magnitude, threshold_value, 255, cv2.THRESH_BINARY)

# 显示二值化后的图像
plt.figure(figsize=(8, 6))
plt.imshow(binary_edge, cmap='gray')
plt.title("Binary Edge Detection")
plt.axis('off')
plt.show()
