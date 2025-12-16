import pandas as pd
import sklearn
from matplotlib import pyplot as plt

import PreProcess_UseModel as pp
from Data.GetData import get

if __name__ == '__main__':

    print(sklearn.__version__)
    if input("pular treino geral?(y/n)").lower() != 'y':
        df1 = get('Data/simulacao-abs.db')
        print(df1.info())
        print( df1[df1.isna().any(axis=1)])
        pp.train_all_models(df1)
    if input("pular teste?(y/n)").lower() != 'y':
        df3 = get('Data/simulacao-abs.db',mode=2)
        y = df3["distancia"].copy()
        df3.pop("distancia")
        result = pd.DataFrame(pp.use_bestModel_for(df3,"mse")).rename(columns={0:"mse"})
        result2 = pd.DataFrame(pp.use_bestModel_for(df3, "r2")).rename(columns={0:"r2"})
        y.reset_index(drop=True, inplace=True)
        print(y)
        r = pd.concat([result,result2,y], axis=1)
        print(r)
        r.plot()
        plt.show()
