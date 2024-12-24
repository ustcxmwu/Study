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
            # text = page.extract_text()
            # print(f"Page {page_num + 1}:")
            # print(text)

            # 如果有表格，提取表格数据
            tables = page.extract_tables()
            res = []
            if tables:
                for table in tables:
                    for row in table:
                        res.append(
                            " ".join(
                                [re.sub(r"[\n/]", "", r) for r in row if r is not None]
                            )
                        )
                        # print(row)
            return " ".join(res)


def get_seller(res):
    # pattern1 = re.compile(r"销售方(?:信息)? *名 ?称[：:]? ?(.+?)(?:统一社会信用代码)? ?纳税人识别号[:：]? ?(\w+)")
    pattern1 = re.compile(
        r"销+售+方+(?:信+息+)? *名+ ?称+ ?[：:]* ?(.+?)(?:统+一+社+会+信+用+代+码+)? ?纳+税+人+识+别+号+ ?[:：]* ?(\w+)",
        flags=re.ASCII,
    )
    match_1 = pattern1.search(res)
    if match_1:
        seller_name = match_1.group(1)
        tax_id = match_1.group(2)
        return seller_name, tax_id
    print(res)
    return None, None


def get_total_amount(res):
    pattern = r"价税合计.*?([\d,.]+)"
    match = re.search(pattern, res)
    if match:
        return match.group(1)
    else:
        return None


def main(month="10月"):
    root = Path("/Users/wuxiaomin/@个人/炎魂网络/@报销/")
    total = 0
    stat_info = defaultdict(list)
    for f in root.rglob(f"{month}/*.pdf"):
        res = plumber_read_text(f)
        seller, seller_id = get_seller(res)
        stat_info["销售方"].append(seller)
        stat_info["销售方纳税人识别号"].append(seller_id)
        amount = get_total_amount(res)
        if amount is None:
            print(f)
            print(res)
            stat_info["金额"].append(0)
        else:
            print(seller, seller_id, amount)
            stat_info["金额"].append(float(amount))
            total += float(amount)
            f.rename(f"{f.parent}/{amount}.pdf")
    print(f"合计金额: {total:.2f}")
    stat_info["销售方"].append("合计")
    stat_info["销售方纳税人识别号"].append("")
    stat_info["金额"].append(total)
    pd.DataFrame(stat_info).to_excel(root / f"{month}.xlsx", index=False)
    # month.rename(f"{month.parent}/{month.stem}_{total:.2f}")


if __name__ == "__main__":
    main(month="12月")
