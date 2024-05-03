#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   drission_eastmoney.py
@Time    :   2024-04-15 15:20
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""

import ddddocr
from DrissionPage import ChromiumOptions, ChromiumPage
from PIL import Image


def binaryzation(image_path, threshold):
    img = Image.open(image_path).convert('L')  # 打开并转为灰度图
    img = img.point(lambda x: 255 if x > threshold else 0)  # 根据阈值进行二值化处理
    img.save('binary.png')  # 保存图片


def get_code(image):
    ocr = ddddocr.DdddOcr(show_ad=False)
    with open(image, 'rb') as f:
        img_bytes = f.read()
    res = ocr.classification(img_bytes)
    return res


def xueqiu_login():
    # 初始化Chromium浏览器选项
    co = ChromiumOptions().auto_port()
    # 创建Chromium页面对象
    page = ChromiumPage(co)

    # 登录需要的账号和密码信息
    username = ''
    cryptographic = ''

    # 打开登录页面
    page.get('https://xueqiu.com/')


    page.ele('.newLogin_modal__login__control_2mV').children()[1].click()
    page.ele('@name=username').input("24339963@qq.com")
    page.ele('@name=password').input("wxm1309")
    page.ele('.:iconfont_iconfont_9UW').click()
    page.ele('.newLogin_modal__login__btn_viK  newLogin_btn-active_144').click()


if __name__ == '__main__':
    xueqiu_login()
