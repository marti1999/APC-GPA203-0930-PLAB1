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

import sympy as sp

pd.set_option("display.max_columns", None)

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

database = pd.read_csv('housing.csv', delim_whitespace=True, names=names)

#ESTA NO ES NUESTRA FUNCION DE COSTE
func = lambda th: np.sin(1 / 2 * th[0] **2 - 1 / 4 * th[1] ** 2 + 3) * np.cos(2 * th[0] + 1 - np.e** th[1])

res = 100
_A = np.linspace(-2, 2, res)
_B = np.linspace(-2, 2, res)

_Cost = np.zeros((res, res))

for ia, a in enumerate(_A):
    for ib, b in enumerate(_B):
        _Cost[ib, ia] = func([a, b])

plt.contourf(_A, _B, _Cost, 100)
plt.colorbar()

Theta = np.random.rand(2) * 4 - 2

_T = np.copy(Theta)

alpha = 0.01
lr = 0.0001

plt.plot(Theta[0], Theta[1], "o", c="white")

gr = np.zeros(2)
nRep = 100000

for m in range(nRep):
    for it, th in enumerate(Theta):
        _T = np.copy(Theta)
        _T[it] = _T[it] + alpha
        d = (func(_T) - func(Theta)) / alpha
        gr[it] = d

    Theta = Theta - lr *  gr
    if (m % 100 == 0):
        plt.plot(Theta[0], Theta[1], ".", c="white")

plt.plot(Theta[0], Theta[1], "o", c="black")
plt.show()