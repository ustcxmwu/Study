#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   reportlib_client.py
@Time    :   2025-11-19 15:05
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2025, Wu Xiaomin
@Desc    :
"""

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def dataframe_to_pdf_table(df, filename="dataframe_output.pdf", title="Pandas DataFrame Report"):
    """
    将 pandas DataFrame 转换为 reportlab Table 并输出为 PDF 文件。

    Args:
        df (pd.DataFrame): 要输出的 pandas DataFrame。
        filename (str): 生成的 PDF 文件名。
        title (str): PDF 文件的标题。
    """

    # 1. 设置文档模板
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    Story = []
    styles = getSampleStyleSheet()

    # 添加标题
    Story.append(Paragraph(f"## {title} ##", styles["h1"]))
    Story.append(Spacer(1, 12))

    # 2. 转换数据
    # 获取列名作为表格的表头
    header = [str(col) for col in df.columns]

    # 将 DataFrame 的数据转换为列表的列表（每一行是一个内部列表）
    data = [header] + df.values.tolist()

    # 3. 创建表格对象
    table = Table(data)

    # 4. 定义表格样式
    style = TableStyle(
        [
            # 整体边框
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            # 表头背景和对齐
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            # 数据行样式
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            # 隔行换色（可选，让表格更易读）
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
            # 循环设置隔行换色（更精确）
            # for i in range(1, len(data)):
            #     bg_color = colors.lightgrey if i % 2 == 0 else colors.white
            #     style.add('BACKGROUND', (0, i), (-1, i), bg_color)
        ]
    )

    table.setStyle(style)

    # 5. 构建文档
    Story.append(table)
    doc.build(Story)

    print(f"✅ PDF 文件已成功生成：{filename}")


if __name__ == '__main__':

    # --- 示例使用 ---
    # 创建一个示例 DataFrame
    data = {
        "Product": ["Apple", "Banana", "Cherry", "Date", "Elderberry"],
        "Quantity": [150, 200, 50, 120, 80],
        "Price": [0.50, 0.30, 1.20, 0.75, 1.50],
        "Total_Sales": [75.00, 60.00, 60.00, 90.00, 120.00],
    }
    df_example = pd.DataFrame(data)

    # 调用函数生成 PDF
    dataframe_to_pdf_table(df_example, filename="Sales_Report.pdf", title="Monthly Sales Data")
