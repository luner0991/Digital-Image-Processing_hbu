import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_rice = cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)
image_cameraman = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)


# 罗伯茨梯度算子
def roberts_gradient(image):
    # 创建罗伯茨梯度算子（水平和垂直）
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    # 使用卷积计算梯度
    grad_x = cv2.filter2D(image.astype(np.float32), -1, kernel_x)
    grad_y = cv2.filter2D(image.astype(np.float32), -1, kernel_y)

    # 计算梯度幅值
    gradient = cv2.magnitude(grad_x, grad_y)
    return gradient


# 锐化处理方法
def sharpen_image(image, method="classic"):
    if method == "classic":
        # 经典锐化：图像加上边缘检测结果
        gradient = roberts_gradient(image)
        sharpened = cv2.add(image.astype(np.uint8), gradient.astype(np.uint8))
        return sharpened

    elif method == "enhance_edges":
        # 增强边缘对比：加大梯度幅度，突出边缘
        gradient = roberts_gradient(image)
        gradient = np.clip(gradient * 2, 0, 255)  # 增强幅度
        sharpened = cv2.add(image.astype(np.uint8), gradient.astype(np.uint8))
        return sharpened

    elif method == "high_pass_filter":
        # 高通滤波锐化：用原图减去低通滤波结果得到高频
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        high_pass = image - blurred
        sharpened = cv2.add(image.astype(np.uint8), high_pass.astype(np.uint8))
        return sharpened

    elif method == "custom_sharpen":
        # 自定义锐化：结合边缘和原图来进行锐化
        gradient = roberts_gradient(image)
        sharpened = image + 0.5 * gradient.astype(np.uint8)
        sharpened = np.clip(sharpened, 0, 255)  # 防止溢出
        return sharpened


# 对两张图像进行锐化处理
methods = ["classic", "enhance_edges", "high_pass_filter", "custom_sharpen"]
results_rice = [sharpen_image(image_rice, method) for method in methods]
results_cameraman = [sharpen_image(image_cameraman, method) for method in methods]

# 显示结果
fig, axes = plt.subplots(2, 5, figsize=(15, 8))

# 显示原图像
axes[0, 0].imshow(image_rice, cmap='gray')
axes[0, 0].set_title('Rice Original')
axes[0, 0].axis('off')

axes[1, 0].imshow(image_cameraman, cmap='gray')
axes[1, 0].set_title('Cameraman Original')
axes[1, 0].axis('off')

# 显示四种锐化结果
for i, method in enumerate(methods):
    axes[0, i + 1].imshow(results_rice[i], cmap='gray')
    axes[0, i + 1].set_title(f'Rice {method}')
    axes[0, i + 1].axis('off')

    axes[1, i + 1].imshow(results_cameraman[i], cmap='gray')
    axes[1, i + 1].set_title(f'Cameraman {method}')
    axes[1, i + 1].axis('off')

plt.tight_layout()
plt.show()
