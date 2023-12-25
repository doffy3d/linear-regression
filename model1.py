import numpy as np
from pandas import read_csv
from sklearn.preprocessing import PolynomialFeatures
from klasa import *
import matplotlib.pyplot as plt


#ucitavanje podataka
data = np.array(read_csv('data.csv', header = None))
X = data[:, :-1]
y = data[:, -1:].reshape(-1)

X = polyExp(standardizacija(X))

#podela na trening i test setove
test_set_size = int(0.20 * X.shape[0])
X_train = X[:-test_set_size, :]
y_train = y[:-test_set_size]

X_test = X[-test_set_size:,:]
y_test = y[-test_set_size:]

val_mean = []
val_std = []
train_mean = []
train_std = []

lam = [0.01, 0.1, 1, 10, 100, 200]
#lam = np.linspace(0,1000,10)

for i in lam:
    #X = standardizacija(polyExpand(X_train[:, 1:], degree = 2, bias = True), bias_included = True)
    model = LinearnaRegresija()
    score = crossValidation(X_train, y_train, model, aux = i)
    val_mean.append(score[0])
    val_std.append(score[1])
    train_mean.append(score[2])
    train_std.append(score[3])

osa = lam
val_mean = np.array(val_mean)
val_std = np.array(val_std)
train_mean = np.array(train_mean)
train_std = np.array(train_std)

#ovim delom koda sam hteo da pronadjem optimalno lambda
mean_score = []
mean_score.append(np.mean(score))
best_lambda = lam[np.argmin(mean_score)]
print("optimalno lambda:", best_lambda)

#procena vrednosti korena srednje kvadratne greske modela
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print("RMSE:", rmse)

#crtanje grafika
plt.figure()
plt.title("Validaciona kriva")
plt.xlabel("Lambda")
plt.ylabel("Skor")
plt.plot(osa, train_mean, label="Trening", color="orange")
plt.fill_between(osa, train_mean - train_std, train_mean + train_std, alpha=0.2, color="orange")
plt.plot(osa, val_mean, label="Validacija", color="navy")
plt.fill_between(osa, val_mean - val_std, val_mean + val_std, alpha=0.2, color="navy")
plt.legend(loc="best")
plt.show()