import cv2
import numpy as np
import matplotlib.pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

# 读取彩色图像
image = cv2.imread('rose.jpg')  # 彩色图像

# 将彩色图像转化为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
w = gray_image.shape[0]
h =gray_image.shape[1]
print(w,h)
# 设置采样因子和量化级数
factor = 4
quantization_levels = 16

# 采样处理
sampled_image = gray_image[::factor, ::factor]

# 量化处理
step = 256 // quantization_levels
quantized_image = (sampled_image // step) * step

# 创建子图显示结果
plt.figure(figsize=(12, 6))

# 显示原始灰度图像
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('原始灰度图像')
plt.axis('off')

# 显示采样后的图像
plt.subplot(1, 3, 2)
plt.imshow(sampled_image, cmap='gray')
plt.title(f'{w//factor}×{h//factor}')
plt.axis('off')

# 显示量化后的图像
plt.subplot(1, 3, 3)
plt.imshow(quantized_image, cmap='gray')
plt.title(f'量化级数: {quantization_levels}')
plt.axis('off')

# 显示图像
plt.tight_layout()
plt.show()
