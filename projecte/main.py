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

pd.set_option("display.max_columns", None)

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

def load_database(path):
    database = pd.read_csv(path, delim_whitespace=True, names=names)
    return database


def marti(database):
    from sklearn.metrics import mean_squared_error, r2_score

    x = database[['NOX']]
    #x = x.values.reshape(1,-1)
    y = database['DIS']
    #y = y.values.reshape(1, -1)

    x_train, x_test, y_train, y_test = train_test_split(x[0], y.values, test_size=0.2, random_state=1845)

    regr = LinearRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    print("mida x_test: ", x_test.shape)
    print("mida y_test: ", y_test.shape)

    print("Coefficients:", regr.coef_)
    print("Intercept:", regr.intercept_)
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))

    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())
    plt.show()

if __name__ == '__main__':
    # Visualitzarem només 3 decimals per mostra
    database=load_database('housing.csv')

    marti(database)
    #test()

    # print("--- ANALYZE DATA ---")
    # print(f"Instances: {database.shape[0]}")
    # print(f"Atributes: {database.shape[1]-1}")
    #
    # print("\n\tTypes of values: ")
    # print(database.dtypes)
    #
    # print("\n\tValues NULL")
    # print(database.info())
    #
    # print("\n\tFirst 5 lines in DDBB")
    # print(database.head())
    #
    # print("\n\tEstatistic numeric atributs:")
    # print(database.describe().T)
    #
    # print("\n\tCorrelation Columns:")
    # print(database.corr())
    #
    #
    # def negatives(val):
    #
    #     if val < 0:
    #         color = 'red'
    #     else:
    #         color = 'black'
    #
    #     return f'color: {color}'
    #
    #
    # def undeline(val):
    #     if val > 0.7 and val < 1:
    #         background = "yellow"
    #     else:
    #         background = "None"
    #
    #     return f'background-color:{background}'
    #
    # # Solo se ve en jupyter
    # database.corr().style.applymap(undeline).applymap(negatives)
    #
    # for i in database.columns:
    #
    #     v= database.iplot(kind="box")
    #     print("hola")
    #
    # """for column in database.columns:
    #     sns.distplot(database[column])
    #     plt.show()
    #
    # plt.figure(figsize=(20, 10))
    # sns.heatmap(database.corr(), annot=True, cmap=plt.cm.CMRmap_r)
    #
    # sns.pairplot(pd.DataFrame(database))"""