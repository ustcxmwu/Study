#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   visual_qa_client.py
@Time    :   2022/9/2 11:14
@Author  :   Wu Xiaomin <xmwu@mail.ustc.edu.cn>
@Version :   1.0
@License :   (C)Copyright 2020-2021, Wu Xiaomin
@Desc    :   
"""
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


if __name__ == "__main__":
    # prepare image + question
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "How many cats are there?"

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
