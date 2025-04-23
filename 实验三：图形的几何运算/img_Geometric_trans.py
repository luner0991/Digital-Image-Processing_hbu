import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库

# 读取图像
image = cv2.imread('slena.jpg')  # 使用cv2.imread函数读取图像文件

# 1. 图像旋转
def rotate_image(image, angle):
    # 获取图像的中心点
    center = (image.shape[1] // 2, image.shape[0] // 2)  # 计算中心点的坐标
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 生成旋转矩阵
    # 应用旋转变换
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))  # 进行旋转变换
    return rotated_image  # 返回旋转后的图像

rotated_image = rotate_image(image, 45)  # 旋转图像45度

# 2. 图像按比例缩放
def resize_image(image, scale_percent):
    # 计算新的宽度和高度
    width = int(image.shape[1] * scale_percent / 100)  # 计算缩放后的宽度
    height = int(image.shape[0] * scale_percent / 100)  # 计算缩放后的高度
    dim = (width, height)  # 新的尺寸元组
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)  # 进行缩放
    return resized_image  # 返回缩放后的图像

resized_image = resize_image(image, 50)  # 按照50%的比例缩放图像

# 3. 图像裁剪
def crop_image(image, start_x, start_y, width, height):
    # 根据给定的起始坐标和宽高进行裁剪
    cropped_image = image[start_y:start_y + height, start_x:start_x + width]  # 裁剪指定区域
    return cropped_image  # 返回裁剪后的图像

cropped_image = crop_image(image, 50, 50, 200, 200)  # 从(50, 50)开始裁剪，宽200高200的区域

# 4. 图像镜像变换
def flip_image(image, mode):
    # 根据模式进行镜像变换
    if mode == 'horizontal':
        flipped_image = cv2.flip(image, 1)  # 水平镜像
    elif mode == 'vertical':
        flipped_image = cv2.flip(image, 0)  # 垂直镜像
    elif mode == 'diagonal':
        flipped_image = cv2.flip(image, -1)  # 对角镜像
    return flipped_image  # 返回镜像后的图像

# 进行各种镜像变换
horizontal_flip = flip_image(image, 'horizontal')  # 水平镜像
vertical_flip = flip_image(image, 'vertical')      # 垂直镜像
diagonal_flip = flip_image(image, 'diagonal')      # 对角镜像

# 5. 图像平移
def translate_image(image, x, y):
    # 创建平移矩阵
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])  # 定义平移矩阵
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))  # 应用平移变换
    return translated_image  # 返回平移后的图像

translated_image = translate_image(image, 100, 50)  # 向右平移100，向下平移50

# 显示结果
cv2.imshow('Original Image', image)  # 显示原始图像
cv2.imshow('Rotated Image', rotated_image)  # 显示旋转后的图像
cv2.imshow('Resized Image', resized_image)  # 显示缩放后的图像
cv2.imshow('Cropped Image', cropped_image)  # 显示裁剪后的图像
cv2.imshow('Horizontal Flip', horizontal_flip)  # 显示水平镜像图像
cv2.imshow('Vertical Flip', vertical_flip)  # 显示垂直镜像图像
cv2.imshow('Diagonal Flip', diagonal_flip)  # 显示对角镜像图像
cv2.imshow('Translated Image', translated_image)  # 显示平移后的图像

cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
