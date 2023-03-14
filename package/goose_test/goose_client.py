#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   goose_client.py
@Time    :   2022/9/7 10:02
@Author  :   Wu Xiaomin <wuxiaomin@pandadastudio.com>
@Version :   1.0
@License :   (C)Copyright 2020-2022, Wu Xiaomin
@Desc    :   
'''
from goose3 import Goose


if __name__ == '__main__':
    config = {}
    config['strict'] = False  # turn of strict exception handling
    config['browser_user_agent'] = 'Mozilla 5.0'  # set the browser agent string
    config['http_timeout'] = 5.05  # set http timeout in seconds
    config["enable_image_fetching"] = True

    with Goose(config) as g:
        article = g.extract("https://forum.xitek.com/thread-1959111-1-1.html")
        # article = g.
        print(article.title)
        print(article.tags)
        print(article.doc)
        print(article.cleaned_text)

