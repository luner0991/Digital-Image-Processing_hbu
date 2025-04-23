'''
@Function:不均匀光照的校正。采用分块处理函数和图像相减函数对图像不均匀光照进行校正
@Author: 刘新媛
@Date: 2024/12/11
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_uneven_illumination(image, block_size=32):
    """
    使用分块处理和图像相减函数对图像进行不均匀光照校正。
    :param image: 输入图像（灰度图）
    :param block_size: 分块大小（默认32）
    :return: 校正后的图像
    """
    # 创建局部均值图（通过分块处理计算图像的局部平均值）
    kernel = (block_size, block_size)
    local_mean = cv2.blur(image, kernel)

    # 校正图像：原始图像减去局部均值图，再通过线性变换调整亮度范围
    corrected = cv2.addWeighted(image, 1.0, local_mean, -1.0, 128)
    return corrected


# 测试图像路径
img_path = "pout.tif"

# 读取灰度图像
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Cannot read image {img_path}")
else:
    # 校正图像
    corrected_image = correct_uneven_illumination(image, block_size=32)

    # 显示原始图像和校正后的图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(corrected_image, cmap='gray')
    axs[1].set_title("Corrected Image")
    axs[1].axis('off')

    # 绘制校正后图像的直方图
    axs[2].hist(corrected_image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axs[2].set_title("Histogram of Corrected Image")
    axs[2].set_xlabel("Pixel Intensity")
    axs[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
