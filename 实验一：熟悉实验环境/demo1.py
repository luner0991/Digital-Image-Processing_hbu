import cv2
import os
import numpy as np
from datetime import datetime

# 1）读入图像并显示
L = cv2.imread('slena.jpg')  # 读取图像
cv2.imshow("image",L)       # 显示图像
cv2.waitKey(0)                # 等待按键关闭窗口
cv2.destroyAllWindows()       # 关闭所有窗口
print("==========================================")
# 2）了解图像文件的信息
file_path = 'slena.jpg'
file_info = {
    "FileName": os.path.basename(file_path),
    "FileModdate": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S'),
    "FileSize": os.path.getsize(file_path),
    "Format": 'JPEG',
    "Width": L.shape[1], # L.shape 返回图像的维度，shape[1] 对应的是图像的宽度（列数）。
    "Height": L.shape[0], # 获取图像的高度。L.shape[0] 对应的是图像的高度（行数）。
    "BitDepth": L.dtype.itemsize * 8, # 位深度指数字图像中每个像素使用的二进制位数，表示该像素能够存储的颜色或灰度级的数量
    "ColorType": '彩色图像' if L.ndim == 3 else '灰度图像', # 判断图像是彩色还是灰度图像。
    # L.ndim 返回数组的维度数, L.ndim == 3，则说明图像有三个维度,表示彩色图像；如果维度数为2，则表示灰度图像。
}

# 打印图像文件的信息
for key, value in file_info.items():
    print(f"{key}: {value}")
print("==========================================")
# 3）显示像素信息，计算像素的平均值和标准差
y, x = 100, 100
pixel_value = L[y, x]  # 获取指定像素的颜色值
# 打印指定像素的颜色值
print(f"在 ({x}, {y})点的像素值为: {pixel_value}")
# 计算图像的平均值和标准差
mean = L.mean()
stddev = L.std()

# 打印平均值和标准差
print(f"图像像素的平均值是: {mean}")
print(f"图像像素的标准差是： {stddev}")
print("==========================================")
# 4）添加均值为0、方差为0.01的高斯白噪声

def gaussian_noise(image, mean, var):  # 返回添加噪声后的图像
    image = np.array(image / 255, dtype=float)  # 将图像像素值归一化到 [0, 1] 范围内
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 生成高斯噪声，均值为 mean，标准差为 var 的平方根
    out = image + noise  # 将生成的噪声添加到归一化后的图像上
    if out.min() < 0:  # 如果叠加噪声后的图像像素值出现负值
        low_clip = -1.  # 设置最小值裁剪为 -1（适用于 [-1, 1] 范围）
    else:
        low_clip = 0.  # 如果没有负值，设置最小值裁剪为 0（适用于 [0, 1] 范围）
    out = np.clip(out, low_clip, 1.0)  # 将图像像素值裁剪到 [low_clip, 1.0] 范围
    out = np.uint8(out * 255)  # 将像素值恢复到 [0, 255] 范围，并转换为 uint8 类型
    return out  # 返回添加了高斯噪声的图像


image = cv2.imread('slena.jpg')
L1 = gaussian_noise(image, mean=0, var=0.01)
# 显示并保存图片
cv2.imshow('Noisy Image', L1)
# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.imwrite("slena_noisy.png", L1)
# 计算L1的像素总数、平均值、标准差
total_pixels_L1 = L1.size
mean_L1 = L1.mean()
stddev_L1 = L1.std()
# 计算L和L1的相关系数
cor_L1 = np.corrcoef(L.flatten(), L1.flatten())[0, 1]

# 打印统计数据
print(f"噪声方差为0.01时,L1的像素总数是: {total_pixels_L1}")
print(f"噪声方差为0.01时,L1的平均灰度值是: {mean_L1}")
print(f"噪声方差为0.01时,L1的灰度标准差是: {stddev_L1}")
print(f"噪声方差为0.01时,L和L1的相关系数是: {cor_L1}")
# 将噪声方差加至0.1，重新计算相关数据
print("==========================================")
L2 = gaussian_noise(image, mean=0, var=0.1)
# 显示并保存图片
cv2.imshow('Noisy Image', L2)
# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.imwrite("slena_noisy2.png", L2)
total_pixels_L2 = L2.size
mean_L2 = L2.mean()
stddev_L2 = L2.std()
cor_L2 = np.corrcoef(L.flatten(), L2.flatten())[0, 1]
print(f"噪声方差为0.1时,L2的像素总数是: {total_pixels_L2}")
print(f"噪声方差为0.1时，L2的平均灰度值是: {mean_L2}")
print(f"噪声方差为0.1时，L2的灰度标准差是: {stddev_L2}")
print(f"噪声方差为0.1时，L和L2的相关系数是: {cor_L2}")
print("==========================================")
# 5）改变图像的尺寸、旋转图像裁剪图像后再进行第四步的统计
size = (512,512)  # 原图尺寸
L_resized = cv2.resize(L, size)
# 旋转图像
L_rotated = cv2.rotate(L, cv2.ROTATE_90_CLOCKWISE)
# 显示旋转后的图像
cv2.imshow('Rotated Image', L_rotated)
cv2.waitKey(0)
# 裁剪图像，假设我们裁剪从(10, 10)到(400, 400)的区域
x1, y1, x2, y2 = 10, 10, 400, 400
L_cropped = L_rotated[y1:y2, x1:x2]
# 显示裁剪后的图像
cv2.imshow('cropped  Image', L_cropped)
cv2.waitKey(0)
L_new = L_cropped
#进行第四步的操作
L3 = gaussian_noise(L_new, mean=0, var=0.1)
# 显示并保存图片
cv2.imshow('Noisy Image', L3)
# 等待用户按下任意键后关闭窗口
cv2.waitKey(0)
cv2.imwrite("slena_noisy3.png", L3)
total_pixels_L3 = L3.size
mean_L3 = L3.mean()
stddev_L3 = L3.std()

print("变换后：\n")
print(f"噪声方差为0.1时,L3的像素总数是: {total_pixels_L3}")
print(f"噪声方差为0.1时，L3的平均灰度值是: {mean_L3}")
print(f"噪声方差为0.1时，L3的灰度标准差是: {stddev_L3}")

