#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   autoscraper_client.py
@Time    :   2024-01-10 15:53
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""

from autoscraper import AutoScraper


if __name__ == "__main__":
    url = "https://weekly.tw93.fun/posts/160-%E7%88%B1%E5%90%83%E7%82%92%E9%A5%AD"

    wanted_list = ["img"]

    scraper = AutoScraper()

    result = scraper.build(url, wanted_list)
    print(result)
