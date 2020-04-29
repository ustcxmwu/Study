import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


if __name__ == '__main__':
    ipos = pd.read_csv(r'./ipo_data.csv', encoding='latin-1')
    print(ipos)

    ipos = ipos.applymap(lambda x: x if not '$' in str(x) else x.replace('$', ''))
    ipos = ipos.applymap(lambda x: x if not '%' in str(x) else x.replace('%', ''))

    print(ipos)
    print(ipos.info())
