#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   w2v_client.py
@Time    :   2022/9/5 10:17
@Author  :   Wu Xiaomin <wuxiaomin@pandadastudio.com>
@Version :   1.0
@License :   (C)Copyright 2020-2021, Wu Xiaomin
@Desc    :   
'''

# -*-coding:utf8-*-
import gensim
import xlrd
import jieba
import numpy as np


def load_data():
    data = xlrd.open_workbook("voice.xlsx")
    table_1 = data.sheets()[0]
    table_2 = data.sheets()[1]
    nrows_1 = table_1.nrows
    nrows_2 = table_2.nrows
    id_question = {}
    for i in range(nrows_1):
        id_question[i] = table_1.row_values(i)
    for i in range(nrows_2):
        data = table_2.row_values(i)
        for id in id_question.keys():
            if id_question[id][1] == data[0]:
                id_question[id].append(data[1])
    return id_question


def cos(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


def load_glove_vector():
    file = open("vectors.txt", "r", encoding="utf8")
    id_vectors = {}
    for line in file:
        data = line.strip().split(" ")
        id_vectors[data[0]] = data[1:]
    return id_vectors


def distance(vector1, vector2):
    d = 0
    for a, b in zip(vector1, vector2):
        d += (a - b) ** 2
    return d ** 0.5


def gensim_word2vec():
    model = gensim.models.Word2Vec.load("27_QA")
    id_question = load_data()
    for id in id_question.keys():
        words = jieba.cut(id_question[id][0])
        # 100维度数组
        vector = np.zeros((1, 100))
        lenght = 0
        for word in words:
            try:
                vector = vector + model.wv[word]
                lenght = lenght + 1
            except:
                continue
        id_question[id].append(vector / lenght)
    while True:
        new_qusetion = input("请输入测试语句：")
        words = jieba.cut(new_qusetion)
        vector = np.zeros((1, 100))
        lenght = 0
        for word in words:
            try:
                vector = vector + model.wv[word]
                lenght = lenght + 1
            except:
                continue
        vector = vector / lenght
        id_mil = {}
        for id in id_question.keys():
            id_mil[id] = distance(vector[0], id_question[id][3][0])
        id = sorted(id_mil.items(), key=lambda d: d[1])
        # 预约好时间检查可以修改吗？
        print(id_question[id[0][0]][2])


def glove_main():
    id_question_glove = load_glove_vector()
    id_question = load_data()
    for id in id_question.keys():
        words = jieba.cut(id_question[id][0])
        # 100维度数组
        vector = np.zeros((1, 100))[0]
        # print(vector)
        lenght = 0
        for word in words:
            try:
                vector = vector + id_question_glove[word]
                # print(id_question_glove[word])
                lenght = lenght + 1
            except:
                continue
        id_question[id].append(vector / lenght)
    while True:
        new_qusetion = input("请输入测试语句：")
        words = jieba.cut(new_qusetion)
        vector = np.zeros((1, 100))[0]
        lenght = 0
        for word in words:
            try:
                vector = vector + id_question_glove[word]
                lenght = lenght + 1
            except:
                continue
        vector = vector / lenght
        id_mil = {}
        for id in id_question.keys():
            id_mil[id] = distance(vector, id_question[id][3])
        id = sorted(id_mil.items(), key=lambda d: d[1])
        # 预约好时间检查可以修改吗？
        print(id_question[id[0][0]][2])


if __name__ == '__main__':
    gensim_word2vec()
