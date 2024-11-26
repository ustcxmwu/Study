#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   demo3.py
@Time    :   2024-11-05 16:03
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""

import cv2
import numpy as np


def grabcut_segmentation(image_path, rect):
    # 读取图像
    image = cv2.imread(image_path)

    # 初始化 mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # 初始化模式和背景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 应用 GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 拉伸结果并掩码图像
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]

    return segmented_image

if __name__ == '__main__':

    # 使用例子
    rect = (50, 50, 450, 290)  # 初始矩形边界，需根据你的图像进行调整
    segmented_image = grabcut_segmentation('./441730773437.jpg', rect)
    cv2.imwrite('segmented_grabcut.jpg', segmented_image)