import os
import pandas as pd
import requests


if __name__ == '__main__':

    PATH = r'./iris/'
    r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

    with open(os.path.join(PATH, 'iris.data'), 'w') as f:
        f.write(r.text)

    df = pd.read_csv(os.path.join(PATH, 'iris.data'), names=['sepal length', 'sepal width', 'petal length',
                                                           'petal width', 'class'])

    print(df.head())
    print(df)
    print(df['sepal length'])
    print("==============================")
    print(df.iloc[:3, :2])

    print(df['class'].unique())

    print(df.count())
    print(df[df['class']=='Iris-virginica'].count())

    virginica = df[df['class'] == 'Iris-virginica'].reset_index(drop=True)
    print(virginica)

    print(df[(df['class']=='Iris-virginica')&(df['petal width']>2.2)])

    print(df.corr())

    print(df.describe(percentiles=[.20,.40,.80,.90,.95]))