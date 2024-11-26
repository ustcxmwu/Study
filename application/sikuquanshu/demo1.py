#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   demo1.py
@Time    :   2024-11-05 11:33
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw


def demo():
    # 读取图像
    image = cv2.imread("./441730773437.jpg")

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用二值化处理
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个白色背景的空白图层
    merged_layer = np.ones_like(image) * 255

    # 识别文字并合并文字图层
    full_text = ""
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image[y : y + h, x : x + w]

        # OCR 识别文字
        text = pytesseract.image_to_string(roi, lang="chi_sim")
        full_text += text + " "

        # 绘制轮廓到新图层，以红色标记
        cv2.rectangle(merged_layer, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 在合并层上显示识别到的完整文字（可选）
    cv2.putText(
        merged_layer,
        full_text.strip(),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imwrite("merged_layer.png", merged_layer)

    # plt.imshow(merged_layer)
    # plt.title("Merged Layer with Text")
    # plt.axis("off")
    # plt.show()

    # 打印识别到的完整文字
    print("识别到的完整文字：")
    print(full_text)


def demo2():
    # 读取图像
    image_path = "./441730773437.jpg"
    # 读取图像
    image = cv2.imread(image_path)
    h_img, w_img, _ = image.shape

    # 使用Pytesseract获取文字信息
    boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # 创建一个透明图层
    overlay = np.zeros((h_img, w_img, 4), dtype="uint8")

    # 将图像转换为PIL格式以便绘制
    pil_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA))
    draw = ImageDraw.Draw(pil_img)

    # 使用PIL绘制新颜色的文本
    for i in range(len(boxes["level"])):
        x, y, w, h = (
            boxes["left"][i],
            boxes["top"][i],
            boxes["width"][i],
            boxes["height"][i],
        )
        text = boxes["text"][i]
        if text.strip():  # 如果text不为空，则绘制
            # 设置字体和大小, 需要安装系统该字体
            # font = ImageFont.truetype("arial.ttf", 16)
            draw.text((x, y), text, fill=(255, 0, 0) + (255,))  # 颜色 + 不透明度


    # 将PIL图像转换为OpenCV
    result_overlay = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)

    # 组合原始图像和绘制覆盖层
    result_image = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_BGR2BGRA), 1, result_overlay, 0.5, 0
    )

    # 保存结果
    cv2.imwrite("./result_image.jpg", result_image)


if __name__ == "__main__":
    demo2()
