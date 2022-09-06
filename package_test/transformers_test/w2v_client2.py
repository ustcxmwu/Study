#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   w2v_client2.py
@Time    :   2022/9/5 11:52
@Author  :   Wu Xiaomin <wuxiaomin@pandadastudio.com>
@Version :   1.0
@License :   (C)Copyright 2020-2022, Wu Xiaomin
@Desc    :   
'''
from typing import List

import jieba
import numpy as np
from gensim.models import Word2Vec
from scipy import spatial


def get_stopwords(filename):
    with open(filename, mode='r') as f:
        stopwords = [line.strip("\n") for line in f.readlines()]
    return stopwords


def clean_stopwords(sentence: str, stopwords: List[str]):
    result = []
    word_list = jieba.lcut(sentence)
    for w in word_list:
        if w not in stopwords:
            result.append(w)
    return result


def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = jieba.lcut(sentence)
    feature_vec = np.zeros((num_features,), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


if __name__ == '__main__':
    s = ["游戏运行时多次出现闪退现象？", "游戏运行时出现画面异常？"]
    stopwords = get_stopwords("./cn_stopwords.txt")

    sentences = [clean_stopwords(ss, stopwords) for ss in s]
    model = Word2Vec(sentences, vector_size=200, sg=0, window=5, min_count=1, workers=4, epochs=5)

    model.save("word2vec.model")
    model.wv.save_word2vec_format('word2vec.vector', binary=False)

    model = Word2Vec.load('word2vec.model')
    # vector = model.wv['画面']
    # print(vector)
    # sims = model.wv.most_similar('画面', topn=10)
    # print(sims)

    index2word_set = set(model.wv.index_to_key)
    s1_afv = avg_feature_vector('游戏异常', model=model, num_features=200, index2word_set=index2word_set)
    s2_afv = avg_feature_vector('游戏画面', model=model, num_features=200, index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    print(sim)

    # dis = model.wmdistance("游戏画面", "游戏开机")
    # print(dis)
