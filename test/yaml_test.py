import yaml
import os


if __name__=='__main__':
    filePath = os.path.dirname(__file__)
    print(filePath)

    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    print(fileNamePath)

    yamlPath = os.path.join(fileNamePath, 'config.yaml')
    print(yamlPath)

    f = open(yamlPath, 'r', encoding='utf-8')

    cont = f.read()

    x = yaml.load(cont)

    print(type(x))
    print(x)
    print(x['EMAIL'])
    print(type(x['EMAIL']))
    print(x['EMAIL']['Smtp_Server'])
    print(type(x['EMAIL']['Smtp_Server']))
    print(x['DB'])
    print(x['DB']['host'])

    print(x.get('DB').get('host'))

    print(type(x.get('DB')))
    fw = open('test.yaml', 'a', encoding='utf-8')
    yaml.dump(x, fw)

