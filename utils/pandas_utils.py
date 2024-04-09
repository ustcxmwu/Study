#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   pandas_utils.py
@Time    :   2024-04-09 11:04
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   pandas 相关工具类
"""
import os

import pandas as pd
from openpyxl import load_workbook


def summay_excel_report(df: pd.DataFrame, filename: os.PathLike, sheet: str, adjust: bool = True) -> None:
    if filename.exists():
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
    else:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
    if adjust:
        adjust_column_width(filename)


def adjust_column_width(file_name: os.PathLike) -> None:
    wb = load_workbook(filename=file_name)
    for worksheets in wb.sheetnames:
        ws = wb[worksheets]
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            ws.column_dimensions[column_letter].width = adjusted_width
    wb.save(filename=file_name)
