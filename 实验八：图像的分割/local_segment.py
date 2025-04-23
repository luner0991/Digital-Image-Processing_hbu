'''
@Function: local Threshold Segmentation局部阈值分割
@Author: 刘新媛
@Date:2024/12/17
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载图像
image = cv2.imread('Cameraman.tif', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    raise ValueError("图像加载失败，请检查图像路径是否正确！")

# 2. 显示原始图像
plt.figure(figsize=(8, 6))
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()


# 3. 多阈值分割
def multi_threshold_segmentation(image, thresholds):
    segmented_images = []
    for i in range(len(thresholds) + 1):
        if i == 0:
            # 小于第一个阈值的部分
            segmented_image = np.where(image <= thresholds[i], 255, 0).astype(np.uint8)
        elif i == len(thresholds):
            # 大于最大阈值的部分
            segmented_image = np.where(image > thresholds[i - 1], 255, 0).astype(np.uint8)
        else:
            # 阈值区间之间的部分
            segmented_image = np.where((image > thresholds[i - 1]) & (image <= thresholds[i]), 255, 0).astype(np.uint8)
        segmented_images.append(segmented_image)

    return segmented_images


# 设置多个阈值，这里我们假设选择了3个阈值，具体值可以根据图像调整
threshold_values = [80, 120, 160]

# 获取多个二值化图像
segmented_images = multi_threshold_segmentation(image, threshold_values)

# 4. 显示每个阈值分割后的图像
for i, segmented_image in enumerate(segmented_images):
    plt.figure(figsize=(8, 6))
    plt.imshow(segmented_image, cmap='gray')
    plt.title(f"Segmented Image with Threshold {threshold_values[i - 1] if i > 0 else '≤ ' + str(threshold_values[0])}")
    plt.axis('off')
    plt.show()

