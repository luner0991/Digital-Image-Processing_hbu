'''
@Function:对测试图像人为加噪后进行平滑处理。根据噪声的不同，选择不同的去噪方法。
@Author: 刘新媛
@Date: 2024/12/11
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载原始图像
image = cv2.imread('eight.tif', cv2.IMREAD_GRAYSCALE)

# 添加噪声
def add_noise(image, noise_type="gaussian"):
    noisy_image = image.copy()
    if noise_type == "gaussian":
        mean = 0
        sigma = 25
        gaussian_noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.clip(image + gaussian_noise, 0, 255)
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.02
        noisy_image = image.copy()
        # Salt noise
        num_salt = int(np.ceil(amount * image.size * s_vs_p))
        salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255
        # Pepper noise
        num_pepper = int(np.ceil(amount * image.size * (1.0 - s_vs_p)))
        pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    return noisy_image

# 生成加噪图像
noisy_image_gaussian = add_noise(image, noise_type="gaussian")
noisy_image_sp = add_noise(image, noise_type="salt_pepper")

# 应用平滑滤波处理
def smooth_image(image, method="gaussian"):
    if method == "mean":
        return cv2.blur(image, (5, 5))  # 均值滤波
    elif method == "median":
        return cv2.medianBlur(image, 5)  # 中值滤波
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)  # 高斯滤波
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)  # 双边滤波

# 针对不同噪声选择合适的去噪方法
smoothed_gaussian = smooth_image(noisy_image_gaussian, method="gaussian")
smoothed_sp = smooth_image(noisy_image_sp, method="median")

# 显示结果
plt.figure(figsize=(12, 8))

# 显示加噪后的图像
plt.subplot(2, 3, 1)
plt.imshow(noisy_image_gaussian, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(noisy_image_sp, cmap='gray')
plt.title('Salt & Pepper Noise')
plt.axis('off')

# 显示平滑后的图像
plt.subplot(2, 3, 4)
plt.imshow(smoothed_gaussian, cmap='gray')
plt.title('Smoothed with Gaussian Filter')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(smoothed_sp, cmap='gray')
plt.title('Smoothed with Median Filter')
plt.axis('off')

plt.tight_layout()
plt.show()
