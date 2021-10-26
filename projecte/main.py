from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import sympy as sy

pd.set_option("display.max_columns", None)

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

database = pd.read_csv('housing.csv', delim_whitespace=True, names=names)

x = database.RM
y = database.MEDV

#ESTA NO ES NUESTRA FUNCION DE COSTE
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1845)

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

N = len(x_train)

def func(a, b, x_train, y_train, N):
    sum = 0
    for i in range(N):
        sum += ((a + b * x_train[i])-y_train[i])**2
    return sum/ N


gy = 30
gx = -gy
res = 100
_A = np.linspace(gx, gy, res)
_B = np.linspace(gx, gy, res)

_Cost = np.zeros((res, res))

for ia, a in enumerate(_A):
    for ib, b in enumerate(_B):
        _Cost[ib, ia] = func(a, b, x_train, y_train, N)

plt.contourf(_A, _B, _Cost, 100)
plt.colorbar()
Theta = np.random.rand(2) * gy

_T = np.copy(Theta)

alpha = 0.01
lr = 0.01

plt.plot(Theta[0], Theta[1], "o", c="white")

gr = np.zeros(2)
nRep = 10000

for m in range(nRep):
    for it, th in enumerate(Theta):
        _T = np.copy(Theta)
        _T[it] = _T[it] + alpha
        d = (func(_T[0], _T[1], x_train, y_train, N) - func(Theta[0], Theta[1], x_train, y_train, N)) / alpha
        gr[it] = d
    Theta = Theta - lr *  gr
    print("cost: ", func(Theta[0], Theta[1], x_train, y_train, N))
    if (m % 50 == 0):
        plt.plot(Theta[0], Theta[1], ".", c="white")


x = np.linspace(0, np.max(x_train))
plt.plot(Theta[0], Theta[1], "o", c="black")
plt.show()
print(Theta)
plt.plot(x_test, y_test, "o", c="red")
plt.plot(x, Theta[0] + Theta[1] * x, "-", c="blue")
plt.show()
