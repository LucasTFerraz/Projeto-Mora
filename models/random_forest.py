import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

'''def use(df):

    X = df.iloc[:, 1:2].values
    y = df.iloc[:, 2].values

    label_encoder = LabelEncoder()
    x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
    x_numerical = df.select_dtypes(exclude=['object']).values
    x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

    regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    regressor.fit(x, y)

    oob_score = regressor.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')

    predictions = regressor.predict(x)

    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y, predictions)
    print(f'R-squared: {r2}')'''
def get_model():
    with open('model_RF.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_RF.json', 'r') as f:
        scores = json.load(f)
    return model,scores

def save_model(model,scores):
    with open('model_RF.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model_RF.json', 'w') as f:
         json.dump(scores, f, indent=4)

def encode_x(df):
    label_encoder = LabelEncoder()
    x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
    x_numerical = df.select_dtypes(exclude=['object']).values
    x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
    return x


def evaluate(model,df):
    from sklearn.metrics import mean_squared_error, r2_score
    x = encode_x(df)
    y = df["distancia"].values
    oob_score = model.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')

    predictions = model.predict(x)

    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y, predictions)
    print(f'R-squared: {r2}')
    return {"R2": r2, "mse": mse, 'Out-of-Bag': oob_score}

def predict(model,df):
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


def train(df):
    y = df["distancia"].values
    x = encode_x(df)

    model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    model.fit(x, y)

    from sklearn.metrics import mean_squared_error, r2_score

    oob_score = model.oob_score_
    print(f'Out-of-Bag Score: {oob_score}')

    predictions = model.predict(x)

    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y, predictions)
    print(f'R-squared: {r2}')
    return model ,{"R2":r2,"mse":mse,'Out-of-Bag':oob_score}