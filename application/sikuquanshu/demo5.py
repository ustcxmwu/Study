#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   demo5.py
@Time    :   2024-11-05 16:10
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def convert_jpg_to_rgb(image_path, output_path):
    # 读取图像，OpenCV 以 BGR 格式读取
    image = cv2.imread(image_path)
    # 转换为 RGB 格式
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 保存图像 (注意：OpenCV 保存图像用的是 BGR 顺序)
    cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

def load_image(image_path):
    # 使用 PIL 读取图像
    image = Image.open(image_path)
    return image

def preprocess_image(image):
    # 预处理图像
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # 增加batch维度

def segment_image(image_tensor):
    # 加载预训练的 DeepLabv3 模型
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()  # 设置为评估模式

    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    return output

def decode_segmentation(output):
    # 将分割结果解码为类索引
    seg = torch.argmax(output, dim=0).numpy()
    return seg

def visualize_segmentation(image, seg):
    # 可视化分割结果
    labels = np.unique(seg)
    segmented_image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label in labels:
        segmented_image[seg == label] = np.random.randint(0, 255, 3)  # 为每个类随机分配颜色

    # 显示原始图像和分割结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Image")

    plt.show()


if __name__ == '__main__':
    image_path = './441730773437.jpg'
    convert_jpg_to_rgb(image_path, "rgb.jpg")
    image = load_image("rgb.jpg")
    image_tensor = preprocess_image(image)
    output = segment_image(image_tensor)
    segmentation = decode_segmentation(output)
    visualize_segmentation(image, segmentation)