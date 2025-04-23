'''
@Function:利用DCT变换实现图像的编码与压缩 并计算压缩比和保真度（以均方误差来衡量
@Author: liu xinyuan
@date:2024/11/20
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def dct_2d(image):
    """计算2D离散余弦变换"""
    return np.fft.fft2(image)

def idct_2d(dct_image):
    """计算2D逆离散余弦变换"""
    return np.real(np.fft.ifft2(dct_image))

def quantize(dct_coeffs, quality):
    """量化DCT系数"""
    max_val = 50 - quality
    quantization_matrix = np.ones_like(dct_coeffs) * max_val
    quantization_matrix[:8, :8] = 10
    quantized_coeffs = np.round(dct_coeffs / quantization_matrix)
    return quantized_coeffs * quantization_matrix

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not open or find the image")
    return img.astype(np.float32) / 255.0

def display_image(title, image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def calculate_mse(original, compressed):
    """计算均方误差"""
    return np.mean((original - compressed) ** 2)

def calculate_compression_ratio(original_coeffs, quantized_coeffs):
    """计算压缩比"""
    total_coeffs = original_coeffs.size
    non_zero_quantized_coeffs = np.count_nonzero(quantized_coeffs)
    return total_coeffs / non_zero_quantized_coeffs

# 主函数
def main(image_path, quality):
    # 读取图像
    img = read_image(image_path)

    # 应用DCT
    dct_img = dct_2d(img)

    # 量化DCT系数
    dct_img_quantized = quantize(dct_img, quality)

    # 应用逆DCT
    img_reconstructed = idct_2d(dct_img_quantized)

    # 归一化到[0, 1]区间
    img_reconstructed = np.clip(img_reconstructed, 0, 1)

    # 显示原始图像和压缩重构图像
    display_image('Original Image', img)
    display_image(f'Compressed Image (Quality={quality})', img_reconstructed)

    # 计算均方误差
    mse = calculate_mse(img, img_reconstructed)
    print(f"MSE: {mse}")

    # 计算压缩比
    compression_ratio = calculate_compression_ratio(dct_img, dct_img_quantized)
    print(f"Compression Ratio: {compression_ratio}")

# 测试图像
image_path1 = 'cameraman.tif'
image_path2 = 'westconcordorthophoto.png'
quality = 40  # 压缩质量，可以调整

main(image_path1, quality)
main(image_path2, quality)