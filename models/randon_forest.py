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

def test():
    df = pd.read_csv('/content/Position_Salaries.csv')
    df.info()

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
    print(f'R-squared: {r2}')
def get_model():
    with open('model.pkl', 'rb') as f:
        clf2 = pickle.load(f)


def save_model(model):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


def predict():
    pass

def train_test():
    pass