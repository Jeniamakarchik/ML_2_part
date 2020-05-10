import os
import time

import numpy as np
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

def read_data():
    print(f'Loading MNIST data to {os.getcwd()}.')
    mnist = sklearn.datasets.fetch_openml('mnist_784', data_home=os.getcwd(), cache=True)
    X = mnist['data']
    y = mnist['target']

    return X, y


def transform_data(X):
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)
    return X_transformed


class OneVSRest(BaseEstimator, ClassifierMixin):
    def __init__(self, labels):
        self.labels = labels

        self.n_cls = len(labels)
        self.clfs = list()

        self.fitting_time = None
        self.probs = None

    def fit(self, X, y):
        start_time = time.time()
        for label in self.labels:
            print(f'Fitting {label} vs rest.')
            mask = (y.astype(int) == label)
            y_two_classes = mask.astype(int)

            clf = LogisticRegression()
            clf.fit(X, y_two_classes)

            self.clfs.append(clf)
        self.time = time.time() - start_time

    def predict_proba(self, X):
        self.probs = np.zeros((len(X), self.n_cls))

        for ind, clf in enumerate(self.clfs):
            y_prob = clf.predict_proba(X)
            self.probs[:, ind] = y_prob[:, 1]

        return self.probs

    def predict(self, X):
        self.predict_proba(X)
        return self.probs.argmax(axis=1)


class OneVSOne(BaseEstimator, ClassifierMixin):
    def __init__(self, labels):
        self.labels = labels

        self.n_cls = len(labels)
        self.clfs = list()

        self.time = None
        self.probs = None

    def fit(self, X, y):
        start_time = time.time()
        self.clfs = []

        for i in range(self.n_cls):
            i_clf = []
            for j in range(i + 1, self.n_cls):
                print(f'Fitting {i} vs {j}.')
                mask = np.array(y.astype(int) == i) | np.array(y.astype(int) == j)

                X_two_classes = X[mask]
                y_two_classes = y[mask].astype(int)

                y_two_classes[y_two_classes == j] = -1
                y_two_classes[y_two_classes == i] = 1

                clf = LogisticRegression()
                clf.fit(X_two_classes, y_two_classes)

                print(accuracy_score(y_two_classes, clf.predict(X_two_classes)))

                i_clf.append(clf)
            self.clfs.append(i_clf)

        self.time = time.time() - start_time

    def predict(self, X):
        self.probs = np.zeros((len(X), self.n_cls, self.n_cls))

        for i in range(self.n_cls):
            for j in range(i + 1, self.n_cls):
                clf = self.clfs[i][j - i - 1]
                self.probs[:, i, j] = clf.predict(X)
                self.probs[:, j, i] = -self.probs[:, i, j]

        return self.probs.sum(axis=1).argmax(axis=1)


if __name__ == '__main__':
    X, y = read_data()
    X_transformed = transform_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.25, random_state=42, stratify=y)
    # one_vs_rest = OneVSRest(list(range(10)))
    #
    # one_vs_rest.fit(X_train, y_train)
    # y_pred = one_vs_rest.predict(X_test)

    one_vs_one = OneVSOne(list(range(10)))
    one_vs_one.fit(X_train, y_train)
    y_pred = one_vs_one.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Fitting time: {one_vs_one.time}')
    # print(f'Confusion matrix')
    # plt.matshow(confusion_matrix(y_test, y_pred))




