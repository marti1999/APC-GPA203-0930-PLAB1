if __name__ == "__main__":
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
    from sklearn.metrics import mean_squared_error, r2_score
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    pd.set_option("display.max_columns", None)
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']
    database = pd.read_csv('housing.csv', delim_whitespace=True, names=names)
    database.info()
    database.head()
    database.describe().T
    plt.figure(figsize=(8,8))
    sns.boxplot(data=database)
    def negatives(val):
        if val < 0 :
            color = 'red'
        else:
            color = 'black'
        return f'color: {color}'
    def undeline(val):
        if abs(val) > 0.7 and abs(val) < 1:
            background = "yellow"
        else:
            background = "None"
        return f'background-color:{background}'
    database.corr().style.applymap(undeline).applymap(negatives)
    plt.figure(figsize=(12,8), dpi=300)
    sns.heatmap(database.corr(), annot=True, linewidths=.5, cmap='rocket');
    sns.pairplot(database[['INDUS', 'PTRATIO', 'LSTAT', 'MEDV']])
    plt.show()
    plt.figure(figsize=(8,8))
    sns.boxplot(data=database[['INDUS', 'PTRATIO', 'LSTAT', 'MEDV']])
    from sklearn.metrics import mean_squared_error, r2_score
    X = database.drop(columns = ['MEDV'])
    y = database.MEDV
    XColumns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']
    XColumns = ['ZN', 'INDUS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1845)
    columnesOrdenades = []
    for col in XColumns:
        x_tr = X_train[[col]]
        x_te = X_test[[col]]
        x_tr = np.reshape(x_tr,(x_tr.shape[0],1))
        x_te = np.reshape(x_te,(x_te.shape[0],1))
        regr = LinearRegression()
        regr.fit(x_tr, y_train)
        y_pred = regr.predict(x_te)
        print("\n", col)
        print("Coefficients:", regr.coef_)
        print("Intercept:", regr.intercept_)
        print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
        print("Coefficient of determination: ", r2_score(y_test, y_pred))
    from sklearn.metrics import mean_squared_error, r2_score
    X = database.drop(columns = ['MEDV'])
    y = database.MEDV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1845)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c='crimson')
    p1 = 50
    p2 = 0
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    print("Coefficients:", regr.coef_)
    print("Intercept:", regr.intercept_)
    print ('Training MSE: ', mean_squared_error(regr.predict(X_train), y_train))
    print("Test MSE: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))
    X = database.drop(columns = ['CRIM', 'CHAS', 'AGE', 'MEDV'])
    y = database.MEDV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1845)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c='crimson')
    p1 = 50
    p2 = 0
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    print("Coefficients:", regr.coef_)
    print("Intercept:", regr.intercept_)
    print ('\nTraining MSE: ', mean_squared_error(regr.predict(X_train), y_train))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))
    X = database.drop(columns = ['CRIM', 'CHAS', 'AGE', 'MEDV'])
    y = database.MEDV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1845)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c='crimson')
    p1 = 50
    p2 = 0
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    print("Coefficients:", regr.coef_)
    print("Intercept:", regr.intercept_)
    print ('Training MSE: ', mean_squared_error(regr.predict(X_train), y_train))
    print("Test MSE: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    scaledX = scaler.transform(X)
    scaledXdf = pd.DataFrame(scaledX)
    scaledXdf.hist(figsize=(8, 8))
    plt.figure(figsize=(8,8))
    sns.boxplot(data=scaledXdf)
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(scaledXdf, y, test_size=0.2, random_state=1845)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c='crimson')
    p1 = 50
    p2 = 0
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    print("Coefficients:", regr.coef_)
    print("Intercept:", regr.intercept_)
    print ('\nTraining MSE: ', mean_squared_error(regr.predict(X_train), y_train))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale
    from sklearn import model_selection
    from sklearn.model_selection import RepeatedKFold
    pca = PCA()
    X_reduced = pca.fit_transform(scale(X))
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    regr = LinearRegression()
    mse = []
    for i in np.arange(1, 12):
        score = -1*model_selection.cross_val_score(regr,
                   X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)
    plt.plot(mse)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('MSE')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1845)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA()
    pca.fit(X_train_scaled)
    print("PCA number of components: ", pca.n_components_)
    X_train_scaled = pca.transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)
    regr = LinearRegression()
    regr.fit(X_train_scaled, y_train)
    y_pred = regr.predict(X_test_scaled)
    print("Coefficients:", regr.coef_)
    print("Intercept:", regr.intercept_)
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c='crimson')
    p1 = 50
    p2 = 0
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    from sklearn.linear_model import SGDRegressor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1845)
    regr = SGDRegressor(max_iter = 1000,penalty = "elasticnet",loss = 'huber',tol = 1e-3, average = True)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    print("Coefficient of determination: ", r2_score(y_test, y_pred))
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c='crimson')
    p1 = 50
    p2 = 0
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    def regulization(X):
        xmat = X.copy()
        x_mean = np.mean(xmat,axis = 0)
        x_std = np.std(xmat,axis = 0)
        x_norm = (xmat-x_mean)/x_std
        return x_norm
    def bgd(data,alp=0.001,numit=50000):
        # dividim les dades en la del aprenentatge per un lloc (columna objectiu) i totes les altres
        xMat = np.mat(data.iloc[:,:13].values)
        yMat = np.mat(data.iloc[:,13].values).T
        # Estandarizar les dades
        xMat = regulization(xMat)
        yMat = regulization(yMat)
        # agafem les files i les columnes de les caracteristiques
        m,n = xMat.shape
        # creem el vector per posar els valors minims de cada variable
        w = np.zeros((n,1))
        #iterem molts cop per anar fins a sota de tot
        for k in range(numit):
            #obtenim el gradien el multipliquem per alfa i fem al resta amb w així obtenim el descens de gradient
            grad = xMat.T*(xMat*w-yMat)/m
            w -= alp*grad
        return w
    bgd(database)
    pd.set_option("display.max_columns", None)
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']
    database = pd.read_csv('housing.csv', delim_whitespace=True, names=names)
    x = database.RM
    y = database.MEDV
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1845)
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    N = len(x_train)
    def func(a, b, x_train, y_train, N):
        sum = 0
        for i in range(N):
            sum += ((a + b * x_train[i])-y_train[i])**2
        return sum/ N
    gy = 40
    gx = -gy
    res = 100
    _A = np.linspace(gx, gy, res)
    _B = np.linspace(gx, gy, res)
    _Cost = np.zeros((res, res))
    for ia, a in enumerate(_A):
        for ib, b in enumerate(_B):
            _Cost[ib, ia] = func(a, b, x_train, y_train, N)
    plt.contourf(_A, _B, _Cost, 100)
    plt.xlabel("A")
    plt.ylabel("B")
    plt.colorbar()
    Theta = np.random.rand(2) * gy
    print("Starting point: ", Theta)
    _T = np.copy(Theta)
    alpha = 0.01
    lr = 0.02
    plt.plot(Theta[0], Theta[1], "o", c="white")
    gr = np.zeros(2)
    nRep = 5000
    for m in range(nRep):
        for it, th in enumerate(Theta):
            _T = np.copy(Theta)
            _T[it] = _T[it] + alpha
            d = (func(_T[0], _T[1], x_train, y_train, N) - func(Theta[0], Theta[1], x_train, y_train, N)) / alpha
            gr[it] = d
        Theta = Theta - lr *  gr
        if (m % 10 == 0):
            plt.plot(Theta[0], Theta[1], ".", c="white")
    x = np.linspace(3, np.max(x_train))
    plt.plot(Theta[0], Theta[1], "o", c="black")
    plt.show()
    plt.plot(x_test, y_test, "o", c="red")
    plt.xlabel("RM")
    plt.ylabel("MEDV")
    plt.plot(x, Theta[0] + Theta[1] * x, "-", c="blue")
    plt.show()
