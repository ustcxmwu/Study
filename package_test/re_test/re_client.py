#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import re

if __name__ == '__main__':
    line = "boooooooobby123"
    regex_str = ".*?(b.*?b).*"
    mat = re.match(regex_str, line)
    if mat:
        print(mat.group(1))
