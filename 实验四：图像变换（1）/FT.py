import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

#  读取图像
img = cv2.imread('Cameraman.tif', 0)  # 以灰度图像形式读取
#  傅里叶变换
f_transform = np.fft.fft2(img)
f_transform_shifted = np.fft.fftshift(f_transform)  # 移动零频率分量到中心
#  频域图像
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
#  逆傅里叶变换
f_ishift = np.fft.ifftshift(f_transform_shifted)  # 反向移动零频率分量
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
# 显示图像
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
plt.figure(figsize=(12, 8))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('原始图像'), plt.axis('off')
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('频域图像'), plt.axis('off')

plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('逆傅里叶变换图像'), plt.axis('off')

plt.show()
