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


def eastmoney_login():
    # 初始化Chromium浏览器选项
    co = ChromiumOptions().auto_port()
    # 创建Chromium页面对象
    page = ChromiumPage(co)

    # 登录需要的账号和密码信息
    username = ''
    cryptographic = ''

    # 打开登录页面
    page.get('https://passport2.eastmoney.com/pub/login?backurl=https%3A%2F%2Fquote.eastmoney.com%2Fzixuan%2F')

    login_frame = page.get_frame("#frame_login")

    page.get_frame(login_frame).ele('.account title').click()
    page.get_frame(login_frame).ele('@name=login_email').input("13405863289")
    page.get_frame(login_frame).ele('@name=login_password').input("kdwxm1309")
    page.get_frame(login_frame).ele('.:selectbox').check()
    page.get_frame(login_frame).ele('#div_vcode').click()
    page.get_frame(login_frame).ele(".em_fullbg em_show").get_screenshot("./pic.jpg")
    ocr = ddddocr.DdddOcr(show_ad=False)
    with open('./pic.jpg', 'rb') as f:
        img_bytes = f.read()
    res = ocr.classification(img_bytes)
    page.get_frame(login_frame).ele('.em_input').input(res)
    page.get_frame(login_frame).ele('.em_valid_btn em_show').click()
    page.get_frame(login_frame).ele('#btn_login').click()


if __name__ == '__main__':
    # eastmoney_login()
    binaryzation('./pic.jpg', 100)
    print(get_code('./binary.png'))
