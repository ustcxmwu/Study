#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   newspaper_client.py
@Time    :   2022/9/7 15:09
@Author  :   Wu Xiaomin <xmwu@mail.ustc.edu.cn>
@Version :   1.0
@License :   (C)Copyright 2020-2022, Wu Xiaomin
@Desc    :   
"""
from newspaper import Article
import newspaper


if __name__ == "__main__":
    # print(newspaper.languages())
    first_article = Article(url="https://photo.xitek.com/", language="zh")
    # first_article = Article(
    #     url="https://forum.xitek.com/thread-1959111-1-1.html", language="zh"
    # )
    first_article.download()
    first_article.parse()
    print(first_article.images)
