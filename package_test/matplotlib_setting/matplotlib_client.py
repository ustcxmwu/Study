#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import matplotlib


if __name__ == '__main__':

    # 查找字体路径
    print(matplotlib.matplotlib_fname())
    # 查找字体缓存路径
    print(matplotlib.get_cachedir())
