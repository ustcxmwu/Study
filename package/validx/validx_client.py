#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.


from validx import Dict, Str, List, Int, exc, Tuple, Float
import validx


if __name__ == '__main__':
    print(validx.__impl__)
    schema = Dict({"message": Str()})
    data = {"message": "ValidX is cool!"}

    print(schema(data))

    s2 = List(
            Float(min=-12, max=1),
            maxlen=10,
            minlen=3
    )
    data = [-1, 1]
    try:
        s2(data)
    except exc.ValidationError as e:
        err = e

    err.sort()
    print(err)


    a = Int(min=0, max=0,options=[1])
    b = -1
    try:
        a(b)
    except exc.ValidationError as e:
        print(e)
