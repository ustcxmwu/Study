#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
from munch import Munch, DefaultMunch


def main():
    profile = Munch()
    print(isinstance(profile, dict))
    pro = DefaultMunch("undefined", {"name": "xiaomin"})
    print(pro.wu)




if __name__ == "__main__":
    main()