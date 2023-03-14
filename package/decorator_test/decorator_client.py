#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

def user_check(func):
    def wraooedfunc(username, passwd):
        if username == "root" and passwd == "123":
            print("通过认证")
            return func()
        else:
            print("用户名或密码错误")

    return wraooedfunc


@user_check
def origin():
    print("xxxxx")


def new_func(func):
    def wrappedfun(*parts):
        if parts:
            for part in parts:
                print(part)
            return func()
        else:
            print("用户名或密码错误")
            return func()

    return wrappedfun


@new_func
def origin2():
    print("yyyyy")


def decrator(*dargs, **dkargs):
    def wrapper(func):
        def _wrapper(*args, **kargs):
            print("装饰器参数:", dargs, dkargs)
            print("函数参数:", args, kargs)
            return func(*args, **kargs)

        return _wrapper

    return wrapper


@decrator("d1", "d2", a=1, b=2)
def origin3(*args, **kargs):
    print("zzzzz")


if __name__ == "__main__":
    origin("root", "123")
    origin2("root", "123")
    origin3("root", "123")
