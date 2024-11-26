#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   demo6.py
@Time    :   2024-11-05 16:23
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def setup_predictor():
    # 创建配置
    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = (
        "http://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 使用 GPU 或 CPU

    return DefaultPredictor(cfg)

def perform_instance_segmentation(image_path, predictor):
    # 读取图像
    image = cv2.imread(image_path)
    # 使用模型进行推断
    outputs = predictor(image)
    # 获取所需的元数据
    metadata = MetadataCatalog.get("coco_2017_val")
    # 创建可视化工具
    visualizer = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
    # 绘制原图加上分割掩码等信息
    vis_output = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    # 返回处理后的图像
    result_image = vis_output.get_image()[:, :, ::-1]

    return result_image

# 使用示例
if __name__ == "__main__":
    predictor = setup_predictor()
    image_path = './441730773437.jpg'
    segmented_image = perform_instance_segmentation(image_path, predictor)
    cv2.imwrite('segmented_image.jpg', segmented_image)