#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   skimage_test.py
@Time    :   2024-04-10 17:05
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import numpy as np
from skimage.color import rgb2hsv
from PIL import Image


def color():
    image = np.array(Image.open("./t.png").convert("RGB"))
    hsv = rgb2hsv(np.zeros_like(image))
    print(hsv)
    hsv2 = rgb2hsv(np.ones_like(image) * 255)
    print(hsv2)
    image1 = Image.fromarray(np.zeros_like(image), mode="RGB")
    hsv3 = np.array(image1.convert("HSV"))
    print(hsv3)
    image2 = Image.fromarray(np.ones_like(image) * 255, mode="RGB")
    hsv4 = np.array(image2.convert("HSV"))
    print(hsv4)


if __name__ == '__main__':
    color()
