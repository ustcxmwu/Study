from decorator import decorator


@decorator
def user_check(func, username="root", passwd="123", *args, **kwargs):
    if username == "root" and passwd == "123":
        print("通过认证")
        return func(*args, **kwargs)
    else:
        print("用户名或密码错误")


@user_check(username="root")
def origin():
    print(origin)
    print("xxxxx")


def main():
    origin()


if __name__ == "__main__":
    main()