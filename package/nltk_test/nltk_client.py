#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   nltk_client.py
@Time    :   2024-02-26 16:29
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import nltk

nltk.download("wordnet")
from nltk.corpus import wordnet


def get_antonyms(word):
    antonymys = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            for antonymy in lemma.antonyms():
                antonymys.add(antonymy.name())
    return antonymys


if __name__ == "__main__":
    print(get_antonyms("美丽"))
