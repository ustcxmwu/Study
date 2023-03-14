#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import numpy as np
import ray


@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        self.params = np.zeros(dim)

    def get_params(self):
        return self.params

    def update_params(self, grad):
        self.params += grad


def main():
    ray.init()
    ps = ParameterServer.remote(10)

    params_id = ps.get_params.remote()
    print(params_id)

    print(ray.get(params_id))


if __name__ == "__main__":
    main()
