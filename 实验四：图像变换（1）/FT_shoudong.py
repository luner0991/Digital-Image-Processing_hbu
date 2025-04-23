import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
# 生成一个简单的图像（黑色背景上的白色矩形）
rows, cols = 64, 64
img = np.zeros((rows, cols), dtype=np.float32)
img[20:44, 20:44] = 1  # 创建一个白色矩形

# 手动实现二维傅里叶变换
def manual_fft2(image):
    # 获取图像的大小
    rows, cols = image.shape
    # 创建输出数组
    f_transform = np.zeros((rows, cols), dtype=complex)
    # 计算傅里叶变换
    for u in range(rows):
        for v in range(cols):
            sum_val = 0
            for x in range(rows):
                for y in range(cols):
                    exponent = -2j * np.pi * (u * x / rows + v * y / cols)
                    sum_val += image[x, y] * np.exp(exponent)
            f_transform[u, v] = sum_val
    return f_transform

# 计算傅里叶变换
f_transform = manual_fft2(img)
# 移动零频率分量到中心
f_transform_shifted = np.fft.fftshift(f_transform)  # 直接调用 numpy 进行平移操作
# 计算幅度谱
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)  # 加1以避免对数的负无穷大

# 显示图像
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('生成的图像')
plt.axis('off')

plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('频域图像')
plt.axis('off')

plt.tight_layout()
plt.show()
