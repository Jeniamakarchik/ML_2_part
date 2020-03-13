import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from random import randint


param_grid = {
    'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
    'n_iter': [100, 200, 500, 1000, 2000, 3000, 5000],
    'alpha': [0.01, 0.001, 0.0001, 0.0],
    'beta': [0.9, 0.99, 0.999],
    'gamma': [0.9, 0.99, 0.999]
}


def prepare_data(X, y, class_name):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = (y == class_name)
    return X.astype(float), np.array(y)


class GDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, type_='stochastic', learning_rate=0.01, n_iter=1000, alpha=0.01,
                 optimization=None, beta=0.9, gamma=0.9, eps=1.0e-9):
        self.type_ = type_
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha

        self.weights = None
        self.coeffs_ = None

        # add optimization
        self.optimization = optimization
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

        self.func_router = {
            'adam': self.adam,
            'rmsprop': self.rmsprop,
            'adagrad': self.adagrad,
            'momentum': self.momentum,
            'nesterov_momentum': self.nesterov_momentum
        }

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
        u = np.zeros_like(self.weights[0])

        for k in range(self.n_iter - 1):
            if self.type_ == 'stochastic':
                ind = randint(0, n_obs - 1)
                x, y = X_train[ind, :], y_train[ind]

            w = self.weights[k]

            try:
                u = self.func_router[self.optimization](x, y, u, w)
            except:
                raise Exception('Unknown optimization.')

            self.weights[k+1] = self.weights[k] + u

        half = self.n_iter//2
        self.coeffs_ = self.weights[half:].mean(axis=0)
        return self

    def predict(self, X):
        return self.sigmoid(X.dot(self.coeffs_)) >= 0.5

    def adam(self, x, y, u, w):
        m = np.zeros_like(self.weights[0])
        v = np.zeros_like(self.weights[0])

        grad = self.compute_gradient(x, y, w) + self.alpha * w
        m = self.gamma * m + (1 - self.gamma) * grad
        v = self.beta * v + (1 - self.beta) * (grad ** 2)
        return - self.learning_rate * m / (np.sqrt(v) + self.eps)

    def rmsprop(self, x, y, u, w):
        g = np.zeros_like(self.weights[0])

        grad = self.compute_gradient(x, y, w) + self.alpha * w
        g = self.beta * g + (1 - self.beta) * (grad ** 2)
        return -self.learning_rate * grad / (np.sqrt(g + self.eps))

    def adagrad(self, x, y, u, w):
        g = np.zeros_like(self.weights[0])

        grad = self.compute_gradient(x, y, w) + self.alpha * w
        g += grad ** 2
        return -self.learning_rate * grad / (np.sqrt(g) + self.eps)

    def momentum(self, x, y, u, w):
        grad = self.compute_gradient(x, y, w) + self.alpha * w
        return self.gamma * u - self.learning_rate * grad

    def nesterov_momentum(self, x, y, u, w):
        w += self.gamma * u
        grad = self.compute_gradient(x, y, w) + self.alpha * w
        return self.gamma * u - self.learning_rate * grad

    def plot_iterations(self, max_iterations, X_train, y_train, X_test, y_test):
        self.n_iter = max_iterations
        self.fit(X_train, y_train)

        accuracies = [accuracy_score(y_test, self.sigmoid(X_test.dot(self.weights[i//2:].mean(axis=0))) >= 0.5)
                      for i in max_iterations]

        plt.plot(range(max_iterations), accuracies)


if __name__ == '__main__':
    classes = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')

    train_data = pd.read_csv('train.csv', header=None)
    test_data = pd.read_csv('test.csv', header=None)

    optimizer = [
        {'type_': 'batch', 'optimization': 'adam'},
        {'type_': 'batch', 'optimization': 'rmsprop'},
        {'type_': 'batch', 'optimization': 'adagrad'},
        {'type_': 'stochastic', 'optimization': 'momentum'},
        {'type_': 'stochastic', 'optimization': 'nesterov_momentum'},
    ]

    for cls in classes:
        X_train, y_train = prepare_data(train_data[[0, 1, 2, 3]], train_data[4], cls)
        X_test, y_test = prepare_data(test_data[[0, 1, 2, 3]], test_data[4], cls)

        for opt in optimizer:
            print('***** ***** ***** *****')
            print(f'Target class: {cls}')
            print(f'Optimizer: {opt}')

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = GridSearchCV(GDClassifier(**opt), param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=kf)
            clf.fit(X_train, y_train)

            print(f'Best classifier: {clf.best_estimator_}')
            print(f'Best CV score: {clf.best_score_:.4}')
            print(f'Test score: {accuracy_score(y_test, clf.predict(X_test)):.4}')
            print('***** ***** ***** *****')
            print()

            clf.best_estimator_.plot_iterations()

        plt.title(f"Accuracy for '{cls}' model")
        plt.show()
