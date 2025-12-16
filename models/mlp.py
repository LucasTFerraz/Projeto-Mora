import json

import pandas as pd

import warnings
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


warnings.filterwarnings('ignore')


def get_model(n):
    with open(f'Trained_Models/model_{n}.pkl', 'rb') as f:
        model = pickle.load(f)

    return model

def save_model(model,name,machine,k):
    with open(f'Trained_Models/model_mlp{name}_{machine}_{k}.pkl', 'wb') as f:
        pickle.dump(model, f)


def encode_x(dfx):
    label_encoder = LabelEncoder()
    #x_categorical = dfx.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
    x_numerical = dfx.select_dtypes(exclude=['object']).values
    '''print("CAT")
    print(pd.DataFrame(x_categorical))
    print("NUM")
    print(pd.DataFrame(x_numerical))
    x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values'''
    x = pd.DataFrame(x_numerical).values
    return x


def evaluate(model,df):

    y = df["distancia"].values
    x = encode_x(df.loc[:, df.columns != "distancia"])

    '''oob_score = model.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')'''

    predictions = model.predict(x)

    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y, predictions)
    print(f'R-squared: {r2}')
    return {"R2": r2, "mse": mse}

def predict(model,df):
    try:
        x = encode_x(df.loc[:, df.columns != "distancia"])
    except:
        x = encode_x(df)
    return model.predict(x)


'''def plot_result(df,regressor,y):
    import numpy as np
    
    X = df.i.loc[:, df.columns != 'distancia'].values
    X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)  # Only the first feature
    X_grid = X_grid.reshape(-1, 1)
    X_grid = np.hstack((X_grid, np.zeros((X_grid.shape[0], 2))))  # Pad with zeros

    plt.scatter(X[:, 0], y, color='blue', label="Actual Data")
    plt.plot(X_grid[:, 0], regressor.predict(X_grid), color='green', label="Random Forest Prediction")
    plt.title("Random Forest Regression Results")
    plt.xlabel('Position Level')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()'''


def train(df,activation = 'relu'):
    y = df["distancia"].values
    x = encode_x(df.loc[:, df.columns != "distancia"])
    print(y)
    print(pd.DataFrame(x))
    model = MLPRegressor(activation=activation, random_state=0)

    model.fit(x, y)

    '''oob_score = model.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')'''

    predictions = model.predict(x)

    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y, predictions)
    print(f'R-squared: {r2}')
    return model ,{"R2":r2,"mse":mse}