#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   qa_client.py
@Time    :   2022/9/2 11:22
@Author  :   Wu Xiaomin <wuxiaomin@pandadastudio.com>
@Version :   1.0
@License :   (C)Copyright 2020-2021, Wu Xiaomin
@Desc    :   
'''
import torch
from transformers import DistilBertTokenizer, DistilBertModel

if __name__ == '__main__':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased-distilled-squad')

    # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    question, text = "二级密码如何进行更换？", "<p>亲爱的忍忍，忍忍可以在二级密码界面直接修改二级密码资讯，如果忍忍忘记了原有的二级密码，可以选择通过初始化二级密码来重置现在的二级密码资讯（选择忘记密码-强制关闭即可发起初始化二级密码申请），选择初始化二级密码后，游戏会在5天后初始化二级密码资讯。</p>"

    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    print(outputs)
