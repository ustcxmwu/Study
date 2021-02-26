#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import math

def great_circle(lon1, lat1, lon2, lat2):
    cdef float radius = 2595
    cdef float x = math.pi/180.0
    cdef float a, b, theta, c
    a = (90.0-lat1)*(x)
    b = (90.0-lat2)*(x)
    theta = (lon2-lon1)*(x)
    c = math.acos((math.cos(a)*math.cos(b)) + (math.sin(a)*math.sin(b*math.cos(theta))))
    return radius*c