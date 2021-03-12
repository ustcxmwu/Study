#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.


class Context(object):
    def __init__(self):
        self.input = ""
        self.output = ""


class AbstractExpression():
    def interpret(self, context):
        pass


class Expression(AbstractExpression):
    def interpret(self, context):
        print("terminal interpret")


class NonterminalExpression(AbstractExpression):
    def interpret(self, context):
        print("Nonterminal interpret")


if __name__ == "__main__":
    context = ""
    c = [Expression(), NonterminalExpression(), Expression(), Expression()]
    for a in c:
        a.interpret(context)

