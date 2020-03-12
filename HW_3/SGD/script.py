import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from random import randint


def prepare_data(X, y, class_name):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = (y == class_name)
    return X.astype(float), np.array(y)


class GDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, type_='stochastic', learning_rate=0.01, n_iter=1000, alpha=0.01, eps=1.0e-9):
        self.type_ = type_
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha
        self.eps = eps

        self.weights = None
        self.coeffs_ = None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def compute_gradient(x, y, w):
        grad = np.dot(x.T, GDClassifier.sigmoid(x.dot(w)) - y)
        return grad if len(x.shape) == 1 else grad / len(x)

    def fit(self, X_train, y_train):
        xs, ys = X_train, y_train
        n_obs = X_train.shape[0]
        n_features = X_train.shape[1]

        self.weights = np.zeros((self.n_iter, n_features))

        for k in range(self.n_iter - 1):
            if self.type_ == 'stochastic':
                ind = randint(0, n_obs - 1)
                x, y = xs[ind, :], ys[ind]

            w = self.weights[k]
            grad = self.compute_gradient(x, y, w) + self.alpha*w

            u = -self.learning_rate*grad
            self.weights[k+1] = self.weights[k] + u

        self.coeffs_ = np.mean(self.weights, axis=0)
        return self

    def predict(self, X):
        return self.sigmoid(X.dot(self.coeffs_)) >= 0.5


if __name__ == '__main__':
    classes = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')

    train_data = pd.read_csv('train.csv', header=None)
    test_data = pd.read_csv('test.csv', header=None)

    for cls in classes:
        print(cls)
        X_train, y_train = prepare_data(train_data[[0, 1, 2, 3]], train_data[4], cls)
        X_test, y_test = prepare_data(test_data[[0, 1, 2, 3]], test_data[4], cls)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print(cross_validate(GDClassifier(), X_train, y_train, scoring='accuracy', cv=kf, n_jobs=1))

        # clf = GDClassifier()
        # clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print(accuracy_score(y_test, y_pred))

