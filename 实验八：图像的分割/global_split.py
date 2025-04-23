'''
@Function: Global Threshold Segmentation全局阈值分割
@Author: 刘新媛
@Date:2024/12/17
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载彩色图像
image = cv2.imread('slena.jpg')

# 2. 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. 显示灰度图像的直方图
def plot_histogram(image, title="Histogram"):
    plt.figure(figsize=(8, 6))
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

plot_histogram(gray_image, "Original Gray Image Histogram")

# 4. 尝试不同的阈值进行二值化
def binarize_image(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# 5. 添加零均值的高斯噪声
def add_gaussian_noise(image, mean=0, std=25):
    row, col = image.shape
    gauss = np.random.normal(mean, std, (row, col))
    noisy_image = np.array(image, dtype=np.float32)
    noisy_image += gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# 6. 添加噪声后的图像
noisy_image = add_gaussian_noise(gray_image)

# 显示加噪后的图像和直方图
plot_histogram(noisy_image, "Noisy Gray Image Histogram")
plt.imshow(noisy_image, cmap='gray')
plt.title("Noisy Gray Image")
plt.show()

# 7. 反复调节阈值并显示二值化效果
# 阈值从50到200
threshold_values = [50, 100, 150, 200]

for threshold in threshold_values:
    binary_image = binarize_image(noisy_image, threshold)
    plt.figure(figsize=(8, 6))
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Image with Threshold {threshold}')
    plt.show()
for threshold in threshold_values:
    binary_image = binarize_image(gray_image, threshold)
    plt.figure(figsize=(8, 6))
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Image with Threshold {threshold}')
    plt.show()