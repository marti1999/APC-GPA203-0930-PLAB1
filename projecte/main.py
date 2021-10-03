from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats

pd.set_option("display.max_columns", None)

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']

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
    print(database.isnull().sum())

    print("\n\tFirst 5 lines in DDBB")
    print(database.head())

    print("\n\tEstatistic numeric atributs:")
    print(database.describe().T)

    print("\n\tCorrelation Columns:")
    print(database.corr())
