#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from cerberus import Validator
import yaml
from easydict import EasyDict
from bidict import bidict, namedbidict



if __name__ == '__main__':
    with open("tmm.yml", mode='r') as f:
        tmm = yaml.safe_load(f)
    print(tmm)

    Tmm = namedbidict("Tmm", "team_member", "model")



