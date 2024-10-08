#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   per_cltv.py
@Time    :   2024-08-08 16:06
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import numpy as np
import pandas as pd
import tensorflow as tf


def data_process(timestep=10, maxlen=64):
    df_S = pd.read_csv('./data/sample_data_individual_behavior.csv')
    df_Y = pd.read_csv('./data/sample_data_label.csv')

    churn_behavior_set = list(map(str, [4, 5, 7, 8, 13, 14, 16, 20, 21, 24, 29,
                                        30, 34, 36, 40, 45, 49, 50, 52, 54, 55, 64, 68, 70, 73, 74, 76, 85, 87, 89]))
    payment_behavior_set = list(
        map(str, [1, 5, 25, 26, 29, 35, 44, 46, 48, 52, 55, 56, 70, 78, 81]))

    B = df_S['seq'].apply(lambda x: x.split(
        ',') if pd.notna(x) else []).tolist()
    C = [list([xx for xx in x if xx in churn_behavior_set]) for x in B]
    P = [list([xx for xx in x if xx in payment_behavior_set]) for x in B]

    B = tf.keras.preprocessing.sequence.pad_sequences(sequences=B,
                                                      maxlen=maxlen,
                                                      padding='post')
    C = tf.keras.preprocessing.sequence.pad_sequences(sequences=C,
                                                      maxlen=maxlen,
                                                      padding='post')
    P = tf.keras.preprocessing.sequence.pad_sequences(sequences=P,
                                                      maxlen=maxlen,
                                                      padding='post')
    B = B.reshape(-1, timestep, maxlen)
    C = C.reshape(-1, timestep, maxlen)
    P = P.reshape(-1, timestep, maxlen)

    y1 = df_Y['churn_label'].values.reshape(-1, 1)
    y2 = np.log(df_Y['payment_label'].values + 1).reshape(-1, 1)

    print('B:', B.shape)
    print('C:', C.shape)
    print('P:', P.shape)
    print('y1:', y1.shape, 'y2:', y2.shape)

    return B, C, P, y1, y2


if __name__ == '__main__':
    data_process()