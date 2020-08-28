#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.



if __name__ == '__main__':
    a = [[], []]
    b = [ item is not None for item in a]
    print(all(a))
    print(all(b))

