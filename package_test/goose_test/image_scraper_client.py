#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   image_scraper_client.py
@Time    :   2022/9/7 10:39
@Author  :   Wu Xiaomin <wuxiaomin@pandadastudio.com>
@Version :   1.0
@License :   (C)Copyright 2020-2022, Wu Xiaomin
@Desc    :   
'''
import requests
from bs4 import BeautifulSoup


def getdata(url):
    r = requests.get(url)
    return r.text


if __name__ == '__main__':
    htmldata = getdata("https://forum.xitek.com/thread-1959111-1-1.html")
    soup = BeautifulSoup(htmldata, 'html.parser')
    for item in soup.find_all('img'):
        print(item['src'])