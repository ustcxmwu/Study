#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   omega_config_client.py
@Time    :   2025-09-15 17:16
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2025, Wu Xiaomin
@Desc    :
"""
from pprint import pprint

from omegaconf import OmegaConf


if __name__ == "__main__":
    # 创建一个包含列表的 OmegaConf 对象
    # Create an OmegaConf object with lists
    conf = OmegaConf.create(
        {
            "dataset": {
                "name": "imagenet",
                "train_transforms": ["resize", "random_crop", "random_flip"],
                "val_transforms": ["resize", "center_crop"],
            },
            "model": {"name": "resnet", "layers": [18, 34, 50]},
        }
    )

    # Use the .pretty() method to print
    pprint(OmegaConf.to_container(conf.dataset, resolve=True))
