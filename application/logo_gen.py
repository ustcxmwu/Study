#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   logo_gen.py
@Time    :   2026-09-01 10:23
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import platform
from PIL import Image, ImageDraw, ImageFont


def generate_group_avatar(
    line1: str, line2: str = "", bg_color: str = "#007aff", size: int = 512
) -> Image.Image:
    """生成正方形群聊群图标，支持 macOS 与 Windows 跨平台字体适配、双行文本、自动缩小及粗体。"""
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    max_width = size * 0.85
    font_size = int(size * 0.18)

    def load_font(size_val):
        system_name = platform.system()
        if system_name == "Darwin":  # macOS 常用中文字体
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Supplemental/Hei.ttf",
                "/System/Library/Fonts/STHeiti Light.ttc",
            ]
        else:  # Windows 常用中文字体
            font_paths = ["simhei.ttf", "msyh.ttf", "simsun.ttc"]

        for path in font_paths:
            try:
                return ImageFont.truetype(path, size_val)
            except IOError:
                continue
        return ImageFont.load_default()

    font = load_font(font_size)
    lines = [line1, line2] if line2 else [line1]

    while font_size > 12:
        font = load_font(font_size)
        too_long = False
        for line in lines:
            if not line:
                continue
            bbox = draw.textbbox((0, 0), line, font=font)
            if (bbox[2] - bbox[0]) > max_width:
                too_long = True
                break
        if not too_long:
            break
        font_size -= 2

    padding = int(font_size * 0.3) if len(lines) == 2 else 0

    bbox_sample = draw.textbbox((0, 0), "测试", font=font)
    line_height = bbox_sample[3] - bbox_sample[1]
    total_height = line_height * len(lines) + padding * (len(lines) - 1)
    start_y = (size - total_height) / 2

    for i, line in enumerate(lines):
        if not line:
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = (size - w) / 2
        y = start_y + i * (line_height + padding)

        # 模拟粗体效果
        for dx in (0, 1):
            for dy in (0, 1):
                draw.text((x + dx, y + dy), line, fill="#ffffff", font=font)

    return img


if __name__ == '__main__':
    avatar = generate_group_avatar("群聊", "", bg_color="#007aff", size=512)
    avatar.save("group_avatar.png")