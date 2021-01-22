#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from statsmodels.tsa import stattools


if __name__ == '__main__':
    # a = list(range(10))
    # acfs = stattools.acf(a, nlags=1, fft=False)
    # print(a)
    # print(acfs)
    # a = ["wo", "shi", "wu", "xiao", "min"]
    # import numpy as np
    # for i in range(10):
    #     np.random.seed(1)
    #     b = np.random.choice(a, 4,  replace=False)
    #     print(b)
    #     c = np.random.choice(a, 4,  replace=False)
    #     print(c)
    # a.append("ustc")
    # a.append("duan")
    # print(a)
    # print(b)
    # print(c)
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # plt.rcParams["figure.figsize"] = 6, 8
    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    #
    # x = np.linspace(-3, 3)
    # y = np.random.randn(len(x), 6)
    #
    # fig, axes = plt.subplots(ncols=2, nrows=3 + 1, gridspec_kw={"height_ratios": [0.02, 1, 1, 1]})
    # fig.suptitle('Some long super title for the complete figure',
    #              fontsize=14, fontweight='bold')
    # for i, ax in enumerate(axes.flatten()[2:]):
    #     ax.plot(x, y[:, i], color=colors[i % 6])
    #     ax.set_title("Title {}".format(i + 1))
    #
    # for i, ax in enumerate(axes.flatten()[:2]):
    #     ax.axis("off")
    #     ax.set_title("Columntitle {}".format(i + 1), fontweight='bold')
    #
    # fig.subplots_adjust(hspace=0.5, bottom=0.1)
    # plt.show()

    print(list(range(1, 1)))