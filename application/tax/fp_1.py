#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   fp_1.py
@Time    :   2024-09-27 18:06
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pdfplumber


def plumber_read_text(f):
    with pdfplumber.open(f) as pdf:
        # 遍历每一页
        for page_num, page in enumerate(pdf.pages):
            # 提取文本
            text = page.extract_text()
            # print(f"Page {page_num + 1}:")
            # print(text)

            # 如果有表格，提取表格数据
            tables = page.extract_tables()
            res = []
            if tables:
                for table in tables:
                    for row in table:
                        res.append(" ".join([re.sub(r'[\n/]', "", r) for r in row if r is not None]))
                        # print(row)
            return res


def get_seller(res):
    pattern1 = re.compile(r"销售方信息\s*名称：(.+?)统一社会信用代码纳税人识别号：(\w+)")
    pattern2 = re.compile(r"销售方\s*名\s*称\s*:\s*([^\n纳税人]+)\s*纳税人识别号\s*:\s*(\d+)")
    for text in res:
        match_1 = pattern1.search(text)
        if match_1:
            seller_name = match_1.group(1)
            tax_id = match_1.group(2)
            return seller_name, tax_id
        match_2 = pattern2.search(text)
        if match_2:
            seller_name = match_2.group(1)
            tax_id = match_2.group(2)
            return seller_name, tax_id
    else:
        return None, None


def get_total_amount(res):
    pattern = r'价税合计.*?([\d,.]+)'
    for text in res:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    else:
        return None


def main(month="10月"):
    root = Path("/Users/wuxiaomin/@个人/炎魂网络/@报销/")
    total = 0
    stat_info = defaultdict(list)
    for f in root.rglob("10月/*.pdf"):
        res = plumber_read_text(f)
        seller, seller_id = get_seller(res)
        stat_info["销售方"].append(seller)
        stat_info["销售方纳税人识别号"].append(seller_id)
        amount = get_total_amount(res)
        if amount is None:
            print(f)
            print(res)
        else:
            print(amount)
            stat_info["金额"].append(float(amount))
            total += float(amount)
        f.rename(f"{f.parent}/{amount}.pdf")
    print(f"合计金额: {total:.2f}")
    stat_info["销售方"].append("合计")
    stat_info["销售方纳税人识别号"].append("")
    stat_info["金额"].append(total)
    pd.DataFrame(stat_info).to_excel(root / f"{month}.xlsx", index=False)
    # month.rename(f"{month.parent}/{month.stem}_{total:.2f}")


if __name__ == '__main__':
    main()
