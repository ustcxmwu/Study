#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import objgraph

if __name__ == '__main__':
    x = ["a", "1", [1, 2]]
    objgraph.show_refs([x], filename="test.png")

    import sys
    import array
    a = array.array('b', 'cstring'.encode())
    print(sys.getsizeof(a))
    l = list("cstring")
    print(sys.getsizeof(l))


