#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   d2v_client.py
@Time    :   2022/9/5 14:40
@Author  :   Wu Xiaomin <xmwu@mail.ustc.edu.cn>
@Version :   1.0
@License :   (C)Copyright 2020-2022, Wu Xiaomin
@Desc    :   
"""
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


if __name__ == "__main__":
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

    vector1 = model.infer_vector(["system", "response"])
    vector2 = model.infer_vector(["system", "response"])
    dis = model.wv.wmdistance(["response", "system"], ["system", "response"])
    print(dis)
