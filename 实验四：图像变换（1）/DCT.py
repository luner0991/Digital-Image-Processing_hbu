import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 读取图像（灰度图像）
image = cv2.imread('Lena.tif', cv2.IMREAD_GRAYSCALE)
img_dct = cv2.dct(np.float32(image))
img_dct_log = 20 * np.log(abs(img_dct))
# 逆离散余弦变换，变换图像回至空间域
img_back = cv2.idct(img_dct)
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
# 绘制原始图像、DCT 幅频谱和 IDCT 恢复的图像
plt.figure(figsize=(12, 6))
# 原始图像
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off')

# DCT 幅频谱图像
plt.subplot(132)
plt.imshow(img_dct_log, cmap='gray')
plt.title('频域图像')
plt.axis('off')

# 逆 DCT 恢复的图像
plt.subplot(133)
plt.imshow(img_back, cmap='gray')
plt.title('逆余弦变换图像')
plt.axis('off')

plt.tight_layout()
plt.show()
