from bidict import namedbidict


if __name__ == '__main__':
    CoupleMap = namedbidict('CoupleMap', 'husbands', 'wives')
    famous = CoupleMap({'bill': 'hillary'})
    print(famous.wives_for['bill'])
    print(famous.husbands_for['hillary'])

    famous.wives_for['barack'] = 'michelle'
    print(famous)
    del famous.husbands_for['hillary']
    print(famous)
    print(famous.pop("barack"))
    print(famous)
