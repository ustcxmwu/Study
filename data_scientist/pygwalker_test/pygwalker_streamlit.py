#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   pygwalker_streamlit.py
@Time    :   2023-12-15 14:18
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import pandas as pd
import streamlit as st

# Adjust the width of the Streamlit page
st.set_page_config(page_title="Use Pygwalker In Streamlit", layout="wide")

# Establish communication between pygwalker and streamlit
init_streamlit_comm()

# Add a title
st.title("Use Pygwalker In Streamlit")


# Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of in-process memory.
@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    # df = pd.read_csv(
    #     "https://kanaries-app.s3.ap-northeast-1.amazonaws.com/public-datasets/bike_sharing_dc.csv"
    # )
    df = pd.read_csv("../dataset/titanic.csv")
    # When you need to publish your app to the public, you should set the debug parameter to False to prevent other users from writing to your chart configuration file.
    return StreamlitRenderer(df, spec="./gw_config.json", debug=False)


if __name__ == "__main__":
    renderer = get_pyg_renderer()

    # Render your data exploration interface. Developers can use it to build charts by drag and drop.
    renderer.render_explore()
