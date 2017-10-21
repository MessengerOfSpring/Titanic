import pandas as pd
import numpy as np
from sklearn import tree
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data_train = pd.read_csv("./data/train.csv")
data_test = pd.read_csv("./data/test.csv")
print('data import complete.')

print(data_train.info())

data_train['Sex'] = data_train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
print(data_train.head())
