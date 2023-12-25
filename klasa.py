import numpy as np
from pandas import read_csv
from sklearn.preprocessing import PolynomialFeatures

def polyExp(X, degree = 2, bias = True):
    pf = PolynomialFeatures(degree = 2, include_bias = bias)
    return pf.fit_transform(X)

def standardizacija(X, bias_included = False):
    X = X - np.mean(X, axis = 0)
    #X = X/np.std(X, axis = 0) #izbaci error pa je mozda pametno da se doda neka devijacija
    epsilon = 1e-10  # mala vrednost
    X = X / (np.std(X, axis=0) + epsilon)
    if bias_included:
        X[:, 0] = [1]*X.shape[0]
    return X

def mse(y_hat, y):
    return np.mean((y_hat - y) ** 2)

def crossValidation(X, y, model, aux=None, n_fold=5):
    train_errors = []
    val_errors = []
    size = X.shape[0]
    X = np.random.permutation(X)
    fold_size = size // n_fold
    for i in range(n_fold):
        mask = [False]*size
        val_indices = list(range(i * fold_size, (i + 1) * fold_size+1))
        for i in val_indices:
            mask[i] = True
            X_val = X[mask, :]
            y_val = y[mask]
            mask = [not mask for mask in mask]
            X_train = X[mask, :]
            y_train = y[mask]
            if model.fit:
                model.fit(X_train, y_train, aux)
                y_kapa = model.predict(X_val)
                train_y_kapa = model.predict(X_train)
            else:
                y_kapa = model.predict(X_train, y_train, aux, X_val)
                train_y_kapa = model.predict(X_train, y_train, aux, X_train)

            train_errors.append(mse(y_train, train_y_kapa))
            val_errors.append(mse(y_kapa, y_val))

        return (-np.mean(val_errors), np.std(val_errors), -np.mean(train_errors), np.std(train_errors))


class LinearnaRegresija:
    def __init__(self):
        self.predict = self.predict
        self.fit = self.fit

    def predict(self, X):
        return X @ self.theta

    def fit(self, X, y, lam, num_iter=1000, tol=1E-6):
        self.theta = (np.linalg.pinv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y)
        #self.theta = (np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y)
        #self.theta = (np.linalg.inv(X.T @ X))
        for j in range(num_iter):
            theta_old = self.theta
            for i, j in enumerate(self.theta):
                xi = X[:, i].reshape((X.shape[0], 1))
                yi = (y - X @ self.theta) + xi * self.theta[i]
                deltai = (xi.T @ yi)[0][0]
                if (deltai < -lam):
                    self.theta[i] = (deltai + lam) / ((xi.T @ xi)[0][0])
                elif (deltai > lam):
                    self.theta[i] = (deltai - lam) / ((xi.T @ xi)[0][0])
                else:
                    self.theta[i] = 0
            if max(abs(self.theta - theta_old)) < tol:
                break
