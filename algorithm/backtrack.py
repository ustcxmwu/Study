#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   backtrack.py
@Time    :   2024-10-28 15:50
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""

from typing import List


class BackTrack(object):
    def __init__(self):
        pass

    def backtrack(self, state, choices: List, res: List):
        if self.is_solution(state):
            self.record_solution(state, res)
        else:
            for choice in choices:
                if self.is_valid(state, choice):
                    self.make_choice(state, choice)
                    self.backtrack(state, choices, res)
                    self.undo_choice(state, choice)

    def is_solution(self, state):
        """
        判断是否是解
        Args:
            state ():

        Returns:

        """
        pass

    def record_solution(self, state, res):
        """
        记录解
        Args:
            state ():
            res ():

        Returns:

        """
        pass

    def is_valid(self, state, choice):
        """
        判断选择是否有效
        Args:
            state ():
            choice ():

        Returns:

        """
        pass

    def make_choice(self, state, choice):
        """
        做选择
        Args:
            state ():
            choice ():

        Returns:

        """
        pass

    def undo_choice(self, state, choice):
        """
        撤销
        Args:
            state ():
            choice ():

        Returns:

        """
        pass


if __name__ == "__main__":
    pass
