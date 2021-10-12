from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats


pd.set_option("display.max_columns", None)

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']

def load_database(path):
    database = pd.read_csv(path, delim_whitespace=True, names=names)
    return database

if __name__ == '__main__':
    # Visualitzarem només 3 decimals per mostra
    database=load_database('housing.csv')

    print("--- ANALYZE DATA ---")
    print(f"Instances: {database.shape[0]}")
    print(f"Atributes: {database.shape[1]-1}")

    print("\n\tTypes of values: ")
    print(database.dtypes)

    print("\n\tValues NULL")
    print(database.info())

    print("\n\tFirst 5 lines in DDBB")
    print(database.head())

    print("\n\tEstatistic numeric atributs:")
    print(database.describe().T)

    print("\n\tCorrelation Columns:")
    print(database.corr())


    def negatives(val):

        if val < 0:
            color = 'red'
        else:
            color = 'black'

        return f'color: {color}'


    def undeline(val):
        if val > 0.7 and val < 1:
            background = "yellow"
        else:
            background = "None"

        return f'background-color:{background}'

    # Solo se ve en jupyter
    database.corr().style.applymap(undeline).applymap(negatives)

    """for column in database.columns:
        sns.distplot(database[column])
        plt.show()

    plt.figure(figsize=(20, 10))
    sns.heatmap(database.corr(), annot=True, cmap=plt.cm.CMRmap_r)

    sns.pairplot(pd.DataFrame(database))"""