from sympy import Point, Circle, Ellipse, Point2D
import math

def sym():
    e1 = Ellipse(Point(0, 0), 3, 2)
    line = e1.tangent_lines(Point(3, 0))
    print(line)

    c1 = Circle(Point(0, 0), 3)
    line = c1.tangent_lines(Point(3, 3))
    print(line)

    print(c1.intersection(line[0]))
    print(c1.intersection(line[1]))
    p = c1.intersection(line[1])
    print(p.x)


def angle_btw_points(point_a, point_b):
    change_in_x = point_b[0] - point_a[0]
    change_in_y = point_b[1] - point_a[1]
    return math.atan2(change_in_y, change_in_x) # remove degrees if you want your answer in radians


def get_point_along_length(point, angle, length):
    return point[0] + length * math.cos(angle), point[1] + length * math.sin(angle)



if __name__ == '__main__':

    print("wwww" == "wwww")



