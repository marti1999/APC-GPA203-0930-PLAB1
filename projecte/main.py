if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    import sklearn as sk
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    import scipy.stats
    import seaborn as sns
    import cufflinks as cf
    import plotly.offline
    import ipywidgets as widgets

    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    pd.set_option("display.max_columns", None)

    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    database = pd.read_csv('housing.csv', delim_whitespace=True, names=names)

    #database.info()

    #database.head()

    from sklearn.metrics import mean_squared_error, r2_score

    X = database
    database.drop(columns=['CRIM', 'ZN', 'CHAS', 'NOX', 'AGE', 'DIS', 'RAD', 'B'])


    y = X.MEDV
    X = X.drop(columns='MEDV')


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1845)

    regr = LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)


    X_testValues = X_test.values
    y_testValues = y_test.values

    plt.plot(X_test.values, y_test.values, 'o', alpha=0.5)
    plt.plot(X_test.values, y_pred, 'r', alpha=0.5)
    plt.show()


    print("Coefficients:", regr.coef_)
    print("Intercept:", regr.intercept_)
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))





