import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from random import randint


param_grid = {
    'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
    'n_iter': [100, 200, 500, 1000, 2000, 3000, 5000],
    'alpha': [0.01, 0.001, 0.0001, 0.0]
}


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
        x, y = X_train, y_train
        n_obs = x.shape[0]
        n_features = x.shape[1]

        self.weights = np.zeros((self.n_iter, n_features))

        for k in range(self.n_iter - 1):
            if self.type_ == 'stochastic':
                ind = randint(0, n_obs - 1)
                x, y = X_train[ind, :], y_train[ind]

            w = self.weights[k]
            grad = self.compute_gradient(x, y, w) + self.alpha*w

            u = -self.learning_rate*grad
            self.weights[k+1] = self.weights[k] + u

        self.coeffs_ = self.weights[-10]
        return self

    def predict(self, X):
        return self.sigmoid(X.dot(self.coeffs_)) >= 0.5


if __name__ == '__main__':
    classes = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')

    train_data = pd.read_csv('train.csv', header=None)
    test_data = pd.read_csv('test.csv', header=None)

    for type_ in ['batch', 'stochastic']:
        for cls in classes:
            print('***** ***** ***** *****')
            print(f'Target class: {cls}')
            print(f'Method: {type_}')
            X_train, y_train = prepare_data(train_data[[0, 1, 2, 3]], train_data[4], cls)
            X_test, y_test = prepare_data(test_data[[0, 1, 2, 3]], test_data[4], cls)

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = GridSearchCV(GDClassifier(type_=type_), param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=kf)
            clf.fit(X_train, y_train)
            print(f'Best classifier: {clf.best_estimator_}')
            print(f'Best mean score: {clf.best_score_:.4}')

            # print(cross_validate(GDClassifier(), X_train, y_train, scoring='accuracy', cv=kf))

            # clf = GDClassifier()
            # clf.fit(X_train, y_train)
            # y_pred = clf.predict(X_test)
            # print(accuracy_score(y_test, y_pred))

            print('***** ***** ***** *****')
            print()

