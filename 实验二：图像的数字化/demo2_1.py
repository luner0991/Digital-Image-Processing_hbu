import cv2
import numpy as np
import matplotlib.pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
# 读取图像
image = cv2.imread('Cameraman.tif', cv2.IMREAD_GRAYSCALE) # 以灰度读取
# 图像尺寸
w = image.shape[0]
h =image.shape[1]
print(w,h)
# 定义不同的采样点数的缩小因子（图像中每单位面积上的像素数量，也称为分辨率）
factors = [ 2, 4, 8, 16, 32]

# 创建子图显示结果,宽10高6
plt.figure(figsize=(10, 6))

# 原始图像显示
plt.subplot(2, 3, 1) # 2行3列的图，一共6个子图
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off') # 关闭坐标轴显示

# 对每个采样点数进行处理和显示
for i, factor in enumerate(factors):
    '''
    采样：
    image[::factor, ::factor]
    第一维度 ::factor：表示沿行方向（垂直方向）每隔 factor 行取一个像素。
    第二维度 ::factor：表示沿列方向（水平方向）每隔 factor 列取一个像素。
    '''
    down_image = image[::factor, ::factor]
    # 显示放大后的图像
    plt.subplot(2, 3, i + 2)
    plt.imshow(down_image, cmap='gray')
    plt.title(f'{w//factor}×{h//factor}')
    plt.axis('off')

# 显示图像
plt.tight_layout()
plt.show()
