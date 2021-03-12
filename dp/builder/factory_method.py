#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import xml.etree.ElementTree as etree
import json


class JSONConnector(object):
    
    def __init__(self, filepath):
        self.data = dict()
        with open(filepath, mode='r') as f:
            self.data = json.load(f)

    @property
    def parsed_data(self):
        return self.data


class XMLConnector(object):

    def __init__(self, filepath):
        self.tree = etree.parse(filepath)

    @property
    def parsed_data(self):
        return self.tree


def connection_factory(filepath):
    if filepath.endswith(".json"):
        connector = JSONConnector
    elif filepath.endswith(".xml"):
        connector = XMLConnector
    else:
        raise ValueError("Cannot connect to {}".format(filepath))
    return connector(filepath)


def connect_to(filepath):
    factory = None
    try:
        factory = connection_factory(filepath)
    except ValueError as ve:
        print(ve)
    return factory


def main():
    sqllite_factory = connect_to("data/person.sq3")

    xml_factory = connect_to("data/person.xml")
    print(xml_factory.parsed_data)

    json_factory = connect_to("data/person.json")
    print(json_factory.parsed_data)


if __name__ == "__main__":
    main()

