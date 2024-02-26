#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   scikit_learn_client1.py
@Time    :   2024-01-11 11:07
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# 读取xlsx数据
df = pd.read_excel("your_data.xlsx")

# 定义特征列和目标列
feature_columns = ["Gender", "Class", "Math_Score"]
target_column = "Target"  # 你的目标列

# 划分特征和目标
X = df[feature_columns]
y = df[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 定义列转换器
# 对 'Gender' 使用 LabelEncoder
# 对 'Class' 使用 OneHotEncoder
# 对 'Math_Score' 使用 StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ("gender_encoder", LabelEncoder(), ["Gender"]),
        ("class_encoder", OneHotEncoder(), ["Class"]),
        ("math_scaler", StandardScaler(), ["Math_Score"]),
    ]
)

# 创建 Pipeline 包括预处理和逻辑回归模型
model = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression())])

# 训练模型
model.fit(X_train, y_train)

# 评估模型
accuracy = model.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.2f}")
