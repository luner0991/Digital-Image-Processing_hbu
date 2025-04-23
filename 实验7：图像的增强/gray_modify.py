'''
@Function:图像灰度修正。读入灰度级分布不协调的图像，分析其直方图。根据直方图设计灰度变换表达式，
          调整表达式的参数，直到显示图像的直方图均衡为止。
@Author: 刘新媛
@Date: 2024/12/11
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, title, ax):
    """
    绘制图像的直方图。
    :param image: 输入图像（灰度图）
    :param title: 直方图标题
    :param ax: Matplotlib的轴对象，用于绘制直方图
    """
    # 使用像素值范围 [0, 256] 绘制直方图，ravel 将图像拉平成一维数组
    ax.hist(image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    ax.set_title(title)  # 标题
    ax.set_xlabel("Pixel Intensity")  # 横轴标签：像素强度
    ax.set_ylabel("Frequency")  # 纵轴标签：频率

def apply_gray_transform(image, alpha=1.0, beta=0):
    """
    应用线性灰度变换，用于调整图像对比度和亮度。
    :param image: 输入图像（灰度图）
    :param alpha: 对比度调整系数（默认值为1.0）
    :param beta: 亮度调整系数（默认值为0）
    :return: 调整后的图像
    """
    # convertScaleAbs函数进行线性变换：output = alpha * input + beta
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def histogram_equalization(image):
    """
    对图像执行直方图均衡化。
    :param image: 输入图像（灰度图）
    :return: 均衡化后的图像
    """
    # equalizeHist 实现直方图均衡化
    return cv2.equalizeHist(image)

images = ["pout.tif", "tire.tif"]

# 遍历每个测试图像
for img_path in images:
    # 读取灰度图像
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 显示原始图像及其直方图
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 创建包含三个子图的图形
    axs[0].imshow(image, cmap='gray')  # 在第一个子图中显示原始图像
    axs[0].set_title("Original Image")  # 设置标题
    axs[0].axis('off')  # 隐藏坐标轴
    plot_histogram(image, "Original Histogram", axs[1])  # 在第二个子图中绘制原始图像直方图
    # 应用灰度变换调整对比度和亮度
    adjusted_image = apply_gray_transform(image, alpha=1.2, beta=15)  # 设置alpha和beta的值
    plot_histogram(adjusted_image, "Adjusted Histogram", axs[2])  # 在第三个子图中绘制调整后的直方图
    # 显示第一组结果
    plt.show()

    # 对原始图像进行直方图均衡化
    equalized_image = histogram_equalization(image)
    # 显示原始图像、灰度调整后的图像和均衡化后的图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 创建包含三个子图的图形
    axs[0].imshow(image, cmap='gray')  # 在第一个子图中显示原始图像
    axs[0].set_title("Original Image")  # 设置标题
    axs[0].axis('off')  # 隐藏坐标轴
    axs[1].imshow(adjusted_image, cmap='gray')  # 在第二个子图中显示调整后的图像
    axs[1].set_title("Adjusted Image")  # 设置标题
    axs[1].axis('off')  # 隐藏坐标轴
    axs[2].imshow(equalized_image, cmap='gray')  # 在第三个子图中显示均衡化后的图像
    axs[2].set_title("Equalized Image")  # 设置标题
    axs[2].axis('off')  # 隐藏坐标轴
    # 显示第二组结果
    plt.show()

    # 绘制均衡化后的直方图及对应的图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 创建包含两个子图的图形
    plot_histogram(equalized_image, "Equalized Histogram", axs[0])  # 在第一个子图中绘制均衡化后的直方图
    axs[1].imshow(equalized_image, cmap='gray')  # 在第二个子图中显示均衡化后的图像
    axs[1].set_title("Equalized Image")  # 设置标题
    axs[1].axis('off')  # 隐藏坐标轴
    # 显示第三组结果
    plt.show()
