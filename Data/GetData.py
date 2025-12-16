# loading in modules
import sqlite3

import numpy as np
import pandas as pd

def sumCol(row,letter):
    if ["distancia"] == 0:
        return row[f'{letter}_dif'] + row[f'S_{letter}_dif']
    return row['eri_hispanic']+row['eri_hispanic']
def get(dbfile = 'simulacao-abs.db',mode = 1):
    # creating file path

    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(dbfile)

    # creating cursor
    cur = con.cursor()

    # reading all table names
    table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    print(table_list)
    cur.execute(f'SELECT * FROM {table_list[0][0]}')
    data = pd.DataFrame(cur.fetchall())
    # here is you table list
    #print(data.info())
    data[0] = pd.to_datetime(data[0])
    d = {0:"Time",1:"X_dif",2:"Y_dif",3:"Z_dif",4:"quebrada_minuto",5:"pausada_minuto"}
    data = data.rename(columns=d)
    data['Time'] = pd.to_numeric(pd.to_datetime(data['Time']))

    #print(data.head())
    #data2 = data.rename(columns=d)
    f = np.array(np.where(data['quebrada_minuto'] == 1)).tolist()[0]
    #print(f)
    fails = {f[0]: {"Start":f[0]}}
    l = f[0]
    for x in range(1,len(f)):
        if f[x]-1 != f[x-1]:

            fails[l]["End"] = f[x-1]
            l = f[x]
            fails[l] = {}
            fails[l]["Start"] = f[x]
    else:
        fails[l]["End"] = f[x]
    #print(fails)
    # Be sure to close the connection
    lk = 0
    ls = 0
    distance = []
    for i in fails.keys():
        y = fails[i]["End"]
        x = fails[i]["Start"]
        ls = x - lk
        while lk <= y:
            if lk>= x:
                distance.append(0)
            else:
                distance.append((x-lk)/ls)
            lk+=1
    else:
        print("d  ",len(distance))
        if mode == 2:
            for x in range(len(data)-len(distance)):
                distance.append(distance[x+4])
        data2 = data[data.index < len(distance)]
    #m = np.linalg.norm(np.array(distance))
    data2["distancia"] = distance #10* np.array(distance)/m
    con.close()
    data2.pop("pausada_minuto")
    data2.pop("quebrada_minuto")
    '''if mode == 0:
        data2=(data2-data2.min())/(data2.max()-data2.min())
        
    else:
        for x in ["X_dif","Y_dif","Z_dif"]:
            data2[f"S_{x}"] = data2["X_dif"].shift(1, fill_value=0)
            data2['X_dif'] = data2.apply(sumCol,args=(x[:1]), axis=1)
        for x in ["X_dif", "Y_dif", "Z_dif"]:
            
            data2.pop(f"S_{x}")'''
    #data2 = (data2 - data2.min()) / (data2.max() - data2.min())
    #data2 = (data2-data2.mean())/data2.std()
    print(data2.info())
    if mode == 2:
        data.pop("pausada_minuto")
        data.pop("quebrada_minuto")
        data = data[data.index >= len(data) - len(data) / 10]
        data2 = data2[data2.index >= len(data2) - len(data2) / 10]
        return data2
    return data2

def plot_lines(df):
    from matplotlib import pyplot as plt
    # data2.pop("pausada_minuto")
    df[880:1110].plot(x="Time", kind="line", figsize=(10, 10))

    # display plot
    plt.show()

#get()
#print("END")