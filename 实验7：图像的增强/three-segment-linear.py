'''
@Function:选择合适的转折点，对图像进行三段线性变换增强。
@Author: 刘新媛
@Date: 2024/12/11
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, title):
    """绘制图像直方图"""
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")


def linear_transform(image, p1, p2):
    """三段线性变换增强"""
    output = np.zeros_like(image, dtype=np.uint8)

    # 低灰度区间
    alpha1 = 255 / p1
    beta1 = 0
    mask1 = (image <= p1)
    output[mask1] = (alpha1 * image[mask1] + beta1).astype(np.uint8)

    # 中灰度区间
    alpha2 = 255 / (p2 - p1)
    beta2 = -alpha2 * p1
    mask2 = (image > p1) & (image <= p2)
    output[mask2] = (alpha2 * image[mask2] + beta2).astype(np.uint8)

    # 高灰度区间
    alpha3 = 255 / (255 - p2)
    beta3 = -alpha3 * p2 + 255
    mask3 = (image > p2)
    output[mask3] = (alpha3 * image[mask3] + beta3).astype(np.uint8)

    return output


# 读取图像
image_path = "couple.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Cannot read image {image_path}")
else:
    # 显示原始图像及其直方图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plot_histogram(image, "Original Histogram")
    plt.show()

    # 设置转折点
    p1, p2 = 85, 170

    # 进行三段线性变换
    enhanced_image = linear_transform(image, p1, p2)

    # 显示增强后的图像及其直方图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title("Enhanced Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plot_histogram(enhanced_image, "Enhanced Histogram")
    plt.show()
