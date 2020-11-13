#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from statsmodels.tsa import stattools


if __name__ == '__main__':
    a = list(range(10))
    acfs = stattools.acf(a, nlags=1, fft=False)
    print(a)
    print(acfs)


