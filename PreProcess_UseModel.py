import json
import models.random_forest as rf
import models.mlp as nn_m


def save_model_score(scores):
    with open('model_Scores.json', 'w') as f:
         json.dump(scores, f, indent=4)

def get_model_scores():
    with open('model_Scores.json', 'r') as f:
        return json.load(f)

def data_split(df,ns):
    m = len(df)
    n = int(m*0.8)#int(df/ns)
    n2 = int(m * 0.9)
    n3 = int(m * 0.7)
    d = {70:[df[df.index < n3],df[df.index >= n3]],80:[df[df.index < n],df[df.index >= n]],90:[df[df.index < n2],df[df.index >= n2]]}
    """for x in range(1,n):
        d = """
    return d
def train_all_models(df,machine ="M1"):
    rf_n = [5,10,15,20]
    mlp_a = ['relu','tanh']
    data = data_split(df,5)
    scores = {}
    for k in data.keys():

        for x in rf_n:
            print(f"RF\n\tK = {k}\n\tX = {x}")
            model,score = rf.train(data[k][0],x)
            score2 = rf.evaluate(model,data[k][1])
            scores[f"RF_{x}_{machine}"] = score2
            rf.save_model(model,x,machine,k)
        for x in mlp_a:
            print(f"mlp\n\tK = {k}\n\tX = {x}")
            model,score = nn_m.train(data[k][0],x)
            score2 = nn_m.evaluate(model,data[k][1])
            scores[f"mlp{x}_{machine}"] = score2
            nn_m.save_model(model,x,machine,k)
    save_model_score(scores)

def train_model(df,type="RF",value = 5,machine ="M1"):
    scores = get_model_scores()
    data = data_split(df, 5)
    for k in data.keys():
        if "RF" in type:
            model,score = rf.train(data[k][0],value)
            score2 = rf.evaluate(model,data[k][1])
            scores[f"RF_{value}_{machine}_{k}"] = score2
            rf.save_model(model,value,machine)
        else:#if "mlp" in res:
            model,score = nn_m.train(data[k][0],value)
            score2 = nn_m.evaluate(model,data[k][1])
            scores[f"mlp{value}_{machine}_{k}"] = score2
            nn_m.save_model(model,value,machine)
    save_model_score(scores)

def use_bestModel_for(df,mode,machine ="M1"):
    d = get_model_scores()
    d = {key: d[key] for key in d.keys() if machine in key}

    match(mode.lower()):
        case("mse"):
            d2 = {v["mse"]:k  for k,v in d.items()}
        case("r2"):
            d2 = {v["R2"]:k  for k,v in d.items()}
        case _:
            d2 = {v["R2"]:k  for k, v in d.items()}

    print(d2)
    res = d2[max(d2.keys())]
    print(f"Best model = {res}")
    if "RF" in res:
        model = rf.get_model(res)
        result = rf.predict(model,df)
    else:#if "mlp" in res:
        model = nn_m.get_model(res)
        result = nn_m.predict(model,df)
    return result