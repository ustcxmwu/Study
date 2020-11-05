#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from importlib import import_module

if __name__ == '__main__':
    a = import_module("modulea.func")
    b = import_module("moduleb.func")
    print(a.add(2, 3))
    print(b.add(2, 3))



