from dynaconf import settings


def main():
    print(settings.as_dict())


if __name__ == "__main__":
    main()