#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   conda_to_pip.py
@Time    :   2023-04-13 14:13
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""


if __name__ == '__main__':
    import yaml

    with open("./environment.yml") as file_handle:
        environment_data = yaml.safe_load(file_handle)

    with open("requirements.txt", "w") as file_handle:
        for dependency in environment_data["dependencies"]:
            package_name, package_version = dependency.split("=")
            file_handle.write("{} == {}\n".format(package_name, package_version))