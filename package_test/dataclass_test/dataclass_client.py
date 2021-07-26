from dataclasses import dataclass
from typing import List


@dataclass
class Package:
    _address: []


@dataclass
class Destination:
    _start: str
    _end: str
    _distance: float


@dataclass
class DestinationContainer:
    _package: Package
    _destinations: List[Destination]

    def addPkg(self, param):
        # sure this shouldn't be "self._package.append(param)"?
        self._package = param


if __name__ == '__main__':
    # works
    dc = DestinationContainer(
        Package(['some address']),
        [Destination('s', 'e', 1.0)]
    )
    print(dc)
    # also works
    dc.addPkg(Package(['some other address']))
    print(dc)
