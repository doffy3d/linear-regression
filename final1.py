import numpy as np
from pandas import read_csv
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class LinearnaRegresija:
    def __init__(self):
        self.theta = None

    def predict(self, X):
        return X @ self.theta

    def fit(self, X, y, lam, num_iter=1000, tol=1E-6):
        self.theta = np.linalg.pinv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y

        for j in range(num_iter):
            theta_old = self.theta
            for i, _ in enumerate(self.theta):
                xi = X[:, i].reshape((X.shape[0], 1))
                yi = (y - X @ self.theta) + xi * self.theta[i]
                deltai = (xi.T @ yi)[0][0]
                if deltai < -lam:
                    self.theta[i] = (deltai + lam) / ((xi.T @ xi)[0][0])
                elif deltai > lam:
                    self.theta[i] = (deltai - lam) / ((xi.T @ xi)[0][0])
                else:
                    self.theta[i] = 0
            if max(abs(self.theta - theta_old)) < tol:
                break

# Load data
data = np.array(read_csv('data.csv', header=None))
X = data[:, :-1]
y = data[:, -1:].reshape(-1)

# Standardize and apply polynomial expansion
pf = PolynomialFeatures(degree=2, include_bias=True)
X_poly = pf.fit_transform(X)

# Split into training and test sets
test_set_size = int(0.20 * X_poly.shape[0])
X_train = X_poly[:-test_set_size, :]
y_train = y[:-test_set_size]

X_test = X_poly[-test_set_size:, :]
y_test = y[-test_set_size:]

val_mean = []
val_std = []
train_mean = []
train_std = []

lam = [0.01, 0.1, 1, 10, 100, 200]

for i in lam:
    model = LinearnaRegresija()
    model.fit(X_train, y_train, lam=i)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)
    train_mean.append(np.mean((y_pred_train - y_train) ** 2))
    val_mean.append(np.mean((y_pred_val - y_test) ** 2))

# Find the optimal lambda
best_lambda = lam[np.argmin(val_mean)]
print("Optimal lambda:", best_lambda)



# Train the model with the optimal lambda
final_model = LinearnaRegresija()
final_model.fit(X_train, y_train, lam=best_lambda)

# Evaluacija
y_pred = final_model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

# Plotting
plt.figure()
plt.title("Validaciona kriva")
plt.xlabel("Lambda")
plt.ylabel("Mean Squared Error")
plt.plot(lam, train_mean, label="Training", color="orange")
plt.plot(lam, val_mean, label="Validation", color="navy")
plt.legend(loc="best")
plt.show()
