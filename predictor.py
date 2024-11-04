'''
The goal of this project is to predict house prices 
based on various factors like location, square footage, 
number of rooms, and more. By leveraging machine learning 
models, we aim to build a robust predictor that can assist 
potential buyers, sellers, and real estate professionals in 
understanding the value of a property based on its unique features.

Author - Dibyajyoti Jena
Date - 04/11/2024
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('Bengaluru_House_Data.csv')

print(data.head())
print(data.info())
print(data.describe())

data = data.dropna(axis=1, thresh=0.8*len(data))
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

X = data.drop('price', axis=1)
y = data['price']

X = pd.get_dummies(X, drop_first=True)