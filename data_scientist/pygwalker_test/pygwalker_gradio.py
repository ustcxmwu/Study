#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   pygwalker_gradio.py
@Time    :   2023-12-15 13:51
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
import pygwalker as pyg
import gradio as gr

df = pd.read_csv("../dataset/titanic.csv")

with gr.Blocks() as demo:
    gr.Label("Visual Titanic Data in PyGWalker and Gradio")
    gr.Markdown(
        "This is a data app built with pygwalker and gradio library. You can use drag-and-drop operations to explore the data. Start your analysis now!"
    )
    gr.HTML(
        pyg.walk(dataset=df, spec="./viz-config.json", debug=False, return_html=True)
    )

if __name__ == "__main__":
    demo.launch()
