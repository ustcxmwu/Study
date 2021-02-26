#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.



if __name__ == '__main__':
    import timeit
    lon1, lat1, lon2, lat2 = -72.345, 34.323, -61.823, 54.826
    num = 500000
    t = timeit.Timer("p2.great_circle({}, {}, {}, {})".format(lon1, lat1, lon2, lat2), "import p2")
    print(t.timeit(num))
    t = timeit.Timer("p1.great_circle({}, {}, {}, {})".format(lon1, lat1, lon2, lat2), "import p1")
    print(t.timeit(num))
    t = timeit.Timer("p3.great_circle({}, {}, {}, {})".format(lon1, lat1, lon2, lat2), "import p3")
    print(t.timeit(num))