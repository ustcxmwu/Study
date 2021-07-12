from jaraco.collections import BijectiveMap as bdict


def main():
    m = bdict()
    m["a"] = "b"
    print(m)
    print(m["b"])
    m.pop("a")
    print(m)


if __name__ == "__main__":
    main()