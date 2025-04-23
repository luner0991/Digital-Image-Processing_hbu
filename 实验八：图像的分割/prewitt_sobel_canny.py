'''
@Function: 分别使用prewitt、sobel、canny算子进行图像分割实验
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

# 2. 使用Prewitt算子进行边缘检测
# Prewitt算子的水平和垂直模板
gx_prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)  # 水平梯度
gy_prewitt = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)  # 垂直梯度

# 使用卷积进行滤波
gradient_h_prewitt = cv2.filter2D(image, -1, gx_prewitt)  # 水平梯度
gradient_v_prewitt = cv2.filter2D(image, -1, gy_prewitt)  # 垂直梯度

# 计算梯度的模
gradient_magnitude_prewitt = np.sqrt(gradient_h_prewitt**2 + gradient_v_prewitt**2)
gradient_magnitude_prewitt = np.uint8(np.clip(gradient_magnitude_prewitt, 0, 255))

# 显示Prewitt算子的边缘检测结果
plt.figure(figsize=(8, 6))
plt.imshow(gradient_magnitude_prewitt, cmap='gray')
plt.title("Edge Detection (Prewitt)")
plt.axis('off')
plt.show()

# 2.1 Prewitt 边缘结果的二值化处理
threshold_value =12  # 设置阈值
_, binary_edge_prewitt = cv2.threshold(gradient_magnitude_prewitt, threshold_value, 255, cv2.THRESH_BINARY)

# 显示Prewitt边缘结果的二值化处理
plt.figure(figsize=(8, 6))
plt.imshow(binary_edge_prewitt, cmap='gray')
plt.title("Binary Edge Detection (Prewitt)")
plt.axis('off')
plt.show()

# 3. 使用Sobel算子进行边缘检测
# Sobel算子的水平和垂直模板
gradient_h_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平梯度
gradient_v_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直梯度

# 计算梯度的模
gradient_magnitude_sobel = np.sqrt(gradient_h_sobel**2 + gradient_v_sobel**2)
gradient_magnitude_sobel = np.uint8(np.clip(gradient_magnitude_sobel, 0, 255))

# 显示Sobel算子的边缘检测结果
plt.figure(figsize=(8, 6))
plt.imshow(gradient_magnitude_sobel, cmap='gray')
plt.title("Edge Detection (Sobel)")
plt.axis('off')
plt.show()

# 3.1 Sobel 边缘结果的二值化处理
_, binary_edge_sobel = cv2.threshold(gradient_magnitude_sobel, threshold_value, 255, cv2.THRESH_BINARY)

# 显示Sobel边缘结果的二值化处理
plt.figure(figsize=(8, 6))
plt.imshow(binary_edge_sobel, cmap='gray')
plt.title("Binary Edge Detection (Sobel)")
plt.axis('off')
plt.show()

# 4. 使用Canny算子进行边缘检测
# Canny算子进行边缘检测
canny_edges = cv2.Canny(image, 100, 200)  # 使用100和200作为低阈值和高阈值

# 显示Canny算子的边缘检测结果
plt.figure(figsize=(8, 6))
plt.imshow(canny_edges, cmap='gray')
plt.title("Edge Detection (Canny)")
plt.axis('off')
plt.show()

# 4.1 Canny 边缘结果的二值化处理
# 对Canny检测的结果进行二值化，Canny本身已是二值化图像，但可以调整阈值进行强化或减弱
_, binary_edge_canny = cv2.threshold(canny_edges, 50, 255, cv2.THRESH_BINARY)

# 显示Canny边缘结果的二值化处理
plt.figure(figsize=(8, 6))
plt.imshow(binary_edge_canny, cmap='gray')
plt.title("Binary Edge Detection (Canny)")
plt.axis('off')
plt.show()
