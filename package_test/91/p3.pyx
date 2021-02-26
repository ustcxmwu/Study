#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

cdef extern from "math.h":
    float cosf(float theta)
    float sinf(float theta)
    float acosf(float theta)


def great_circle(float lon1, float lat1, float lon2, float lat2):
    cdef float radius = 2595
    cdef float x = 3.14159265/180.0
    cdef float a, b, theta, c
    a = (90.0-lat1)*(x)
    b = (90.0-lat2)*(x)
    theta = (lon2-lon1)*(x)
    c = acosf((cosf(a)*cosf(b)) + (sinf(a)*sinf(b*cosf(theta))))
    return radius*c