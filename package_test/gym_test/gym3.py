from gym import envs


def main():
    env_names = [spec.id for spec in envs.registry.all()]
    for name in sorted(env_names):
        print(name)


if __name__ == "__main__":
    main()