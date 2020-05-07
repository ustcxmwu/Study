import json



if __name__ == '__main__':
    # data = [{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}]
    #
    # json_str = json.dumps(data)
    # print(json_str)
    # obj = json.loads(json_str)
    # print(obj[0]['a'])
    # if a == 5:
    #     print(True)
    with open("test_nn_reward.json", "r") as f:
        a = json.load(f)
    print(a)
