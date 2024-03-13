#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   nltk_words.py
@Time    :   2024-03-12 17:33
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""

import nltk

nltk.download("words")
from nltk.corpus import wordnet

import synonyms


def check_synonyms(word1, word2):
    synset1 = wordnet.synsets(word1)
    synset2 = wordnet.synsets(word2)
    print(synset1)
    print(synset2)

    for syn1 in synset1:
        for syn2 in synset2:
            if syn1.wup_similarity(syn2) > 0.8:
                return True
    return False


def check_synonyms_1(word1, word2):
    synset1 = synonyms.nearby(word1)
    synset2 = synonyms.nearby(word2)
    print(synset1)
    print(synset2)

    for syn1 in synset1:
        for syn2 in synset2:
            if syn1.wup_similarity(syn2) > 0.8:
                return True
    return False


if __name__ == "__main__":
    print(check_synonyms_1("正装", "西装"))
