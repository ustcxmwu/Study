import json
import xml.etree.ElementTree as etree
from abc import ABCMeta, abstractmethod


class Connector(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, filepath):
        pass

    @abstractmethod
    def parsed_data(self):
        pass


class JSONConnector(Connector):

    def __init__(self, filepath):
        self.data = dict()
        with open(filepath, mode='r') as f:
            self.data = json.load(f)

    def parsed_data(self):
        return self.data


class XMLConnector(Connector):

    def __init__(self, filepath):
        self.tree = etree.parse(filepath)

    def parsed_data(self):
        return self.tree


def connection_factory(filepath) -> Connector:
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
    print(xml_factory.parsed_data())

    json_factory = connect_to("data/person.json")
    print(json_factory.parsed_data())


if __name__ == "__main__":
    main()
