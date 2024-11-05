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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """Loads the dataset, handles missing values, and prepares it for modeling."""
    data = pd.read_csv(file_path)
    print("Data loaded successfully.\n")
    
    print("First 5 rows of the dataset:")
    print(data.head(), "\n")
    print("Dataset Info:")
    print(data.info(), "\n")
    print("Dataset Description:")
    print(data.describe(), "\n")
    
    data = data.dropna(axis=1, thresh=0.8 * len(data))
    print("Columns with more than 20% missing values have been dropped.\n")

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    print("Remaining missing values in numeric columns have been filled with the mean.\n")

    return data


def prepare_features_and_target(data, target_column='price'):
    """Separates features (X) and target (y) from the dataset"""
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X = pd.get_dummies(X, drop_first=True)
    print("Categorical variables have been converted to dummy variables.\n")

    return X, y

def standardize_data(X_train, X_test):
    """Standardizes the training and testing data"""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data has been standardized.\n")

    return X_train, X_test

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Trains a Linear Regresssion model and evaluates it on test set"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed.\n")

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}\n")

    return model, y_pred

def plot_prediction(y_test, y_pred):
    """Plots actual vs. predicted values for visual evaluation of the model."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', color='red', lw=2)
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs. Predicted Sale Prices")
    plt.show()   

file_path = 'Bengaluru_House_Data.csv'
data = load_and_preprocess_data(file_path)

X, y = prepare_features_and_target(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train, X_test = standardize_data(X_train, X_test)

model, y_pred = train_and_evaluate_model(X_train, y_train, X_test, y_test)

plot_prediction(y_test, y_pred)