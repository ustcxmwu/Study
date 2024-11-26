import cv2
import numpy as np


def simple_threshold_segmentation(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值化
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    segmented_image = np.zeros_like(image)
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 3)  # 绘制轮廓为绿色

    return cv2.bitwise_or(image, segmented_image)


if __name__ == '__main__':

    # 使用示例
    segmented_image = simple_threshold_segmentation("./441730773437.jpg")
    cv2.imwrite("segmented_image.jpg", segmented_image)
