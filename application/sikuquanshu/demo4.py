#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   demo4.py
@Time    :   2024-11-05 16:05
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""
from skimage import io, color, filters, segmentation, measure
import numpy as np


def watershed_segmentation(image_path):
    # 读取图像
    image = io.imread(image_path)
    gray = color.rgb2gray(image)

    # 应用高斯模糊
    blurred = filters.gaussian(gray, sigma=1.0)

    # 寻找边缘（梯度）
    elevation_map = filters.sobel(blurred)

    # 使用分水岭算法
    markers = np.zeros_like(gray)
    markers[gray < 0.4] = 1
    markers[gray > 0.6] = 2
    segmentation_labels = segmentation.watershed(elevation_map, markers)

    # 提取轮廓
    segmented_image = color.label2rgb(segmentation_labels, image=image, kind='overlay')

    return segmented_image


if __name__ == '__main__':
    segmented_image = watershed_segmentation('./441730773437.jpg')
    io.imsave('segmented_watershed.jpg', segmented_image)