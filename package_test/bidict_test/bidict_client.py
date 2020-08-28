from bidict import namedbidict


if __name__ == '__main__':
    CoupleMap = namedbidict('CoupleMap', 'husbands', 'wives')
    famous = CoupleMap({'bill': 'hillary'})
    print(famous.wives_for['bill'])
    print(famous.husbands_for['hillary'])

    famous.husbands['barack'] = 'michelle'
    del famous.wives['hillary']
    print(famous)
