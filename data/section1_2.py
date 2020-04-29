import matplotlib.pyplot as plt
import pandas as pd
import os
plt.style.use('ggplot')

import numpy as np
import seaborn as sns
import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def read_iris():
    PATH = r'./iris/'
    df = pd.read_csv(os.path.join(PATH, 'iris.data'), names=['sepal length', 'sepal width', 'petal length',
                                                             'petal width', 'class'])
    return df


def fuc1():
    df = read_iris()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['petal width'], color='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Width', fontsize=12)
    plt.title('Iris Petal Width', fontsize=14, y=1.01)
    plt.show()


def func2():
    df = read_iris()
    fig, ax = plt.subplots(2, 2, figsize=(6, 4))
    ax[0][0].hist(df['petal width'], color='black')
    ax[0][0].set_ylabel('Count', fontsize=12)
    ax[0][0].set_xlabel('Width', fontsize=12)
    ax[0][0].set_title('Iris Petal Width', fontsize=14, y=1.01)

    ax[0][1].hist(df['petal length'], color='black')
    ax[0][1].set_ylabel('Count', fontsize=12)
    ax[0][1].set_xlabel('Length', fontsize=12)
    ax[0][1].set_title('Iris Petal Length', fontsize=14, y=1.01)

    ax[1][0].hist(df['sepal width'], color='black')
    ax[1][0].set_ylabel('Count', fontsize=12)
    ax[1][0].set_xlabel('Width', fontsize=12)
    ax[1][0].set_title('Iris Sepal Width', fontsize=14, y=1.01)

    ax[1][1].hist(df['sepal length'], color='black')
    ax[1][1].set_ylabel('Count', fontsize=12)
    ax[1][1].set_xlabel('Length', fontsize=12)
    ax[1][1].set_title('Iris Sepal Length', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.show()



def func3():
    df = read_iris()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df['petal width'], df['petal length'], color='green')
    ax.set_xlabel('Petal Width')
    ax.set_ylabel('Petal Length')
    ax.set_title('Petal Scatterplot')
    plt.show()


def func4():
    df = read_iris()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(df['petal length'], color='black')
    ax.set_xlabel('Specimen Number')
    ax.set_ylabel('Petal Length')
    ax.set_title('Petal Length Plot')
    plt.show()


def func5():
    df = read_iris()
    fig, ax = plt.subplots(figsize=(6, 6))
    bar_width = .8
    labels = [x for x in df.columns if 'length' in x or 'width' in x]
    ver_y = [df[df['class'] == 'Iris-versicolor'][x].mean() for x in labels]
    vir_y = [df[df['class'] == 'Iris-virginica'][x].mean() for x in labels]
    set_y = [df[df['class'] == 'Iris-setosa'][x].mean() for x in labels]
    x = np.arange(len(labels))
    ax.bar(x, vir_y, bar_width, bottom=set_y, color='darkgrey')
    ax.bar(x, set_y, bar_width, bottom=ver_y, color='white')
    ax.bar(x, ver_y, bar_width, color='black')
    ax.set_xticks(x + (bar_width/2))
    ax.set_xticklabels(labels, rotation=-70, fontsize=12)
    ax.set_title('Mean Feature Measurement By Class', y=1.01)
    ax.legend(['Virginica', 'Setosa', 'Versicolor'])
    plt.show()


def func6():
    df = read_iris()
    sns.pairplot(df, hue='class')
    plt.show()


def func7():
    df = read_iris()
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    sns.set(style='white', palette='muted')
    sns.violinplot(x=df['class'], y=df['sepal length'], ax=ax[0, 0])
    sns.violinplot(x=df['class'], y=df['sepal width'], ax=ax[0, 1])
    sns.violinplot(x=df['class'], y=df['petal length'], ax=ax[1, 0])
    sns.violinplot(x=df['class'], y=df['petal width'], ax=ax[1, 1])
    fig.suptitle('Violin Plots', fontsize=16, y=1.03)
    for i in ax.flat:
        plt.setp(i.get_xticklabels(), rotation=-90)
    fig.tight_layout()
    plt.show()


def func8():
    df = read_iris()
    y = df['sepal length'][:50]
    x = df['sepal width'][:50]
    X = sm.add_constant(x)
    results = sm.OLS(y, X).fit()
    print(results.summary())
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, results.fittedvalues, label='regression line')
    ax.scatter(x, y, label='data point', color='r')
    ax.set_ylabel('Sepal Length')
    ax.set_xlabel('Sepal Width')
    ax.set_title('Setosa Sepal Width vs. Sepal Length', fontsize=14,
                 y=1.02)
    ax.legend(loc=2)
    plt.show()


def func9():
    clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    df = read_iris()
    X = df.iloc[:, :4]
    y = df.iloc[:, 4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    rf = pd.DataFrame(list(zip(y_pred, y_test)), columns=['predicted', 'actual'])
    rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis=1)
    print(rf)
    print(rf['correct'].sum()/rf['correct'].count())


def func10():
    df = read_iris()
    clf = OneVsRestClassifier(SVC(kernel='linear'))
    X = df.iloc[:, :4]
    y = np.array(df.iloc[:, 4]).astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    rf = pd.DataFrame(list(zip(y_pred, y_test)), columns=['predicted', 'actual'])
    rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis=1)
    print(rf)
    print(rf['correct'].sum() / rf['correct'].count())





if __name__ == '__main__':
    func10()

