import cv2
import numpy as np
import matplotlib.pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 读取图像
image = cv2.imread('Cameraman.tif', cv2.IMREAD_GRAYSCALE)  # 以灰度读取

# 图像尺寸
w, h = image.shape
print(w, h)

# 定义不同的量化级数
quantization_levels = [32, 16, 8, 4, 2]

# 创建子图显示结果, 宽10高6
plt.figure(figsize=(10, 6))

# 原始图像显示
plt.subplot(2, 3, 1)  # 2行3列的图，一共6个子图
plt.imshow(image, cmap='gray')
plt.title('原始图像 灰度级为256')
plt.axis('off')  # 关闭坐标轴显示

# 对每个量化级数进行处理和显示
for i, levels in enumerate(quantization_levels):
    # 计算量化步长
    step = 256 // levels
    '''
    量化：
    (image // step) * step
    step 是每个量化级别的步长，将像素值映射到离散的量化级别。
    经过整数除法后，结果是这些量化级别的索引，然后再乘以 step，得到实际的像素值。
    '''
    # 量化处理
    quantized_image = (image // step) * step
    # 显示处理后的图像
    plt.subplot(2, 3, i + 2)
    plt.imshow(quantized_image, cmap='gray')
    plt.title(f'灰度级为: {levels}')
    plt.axis('off')

# 显示图像
plt.tight_layout()
plt.show()
