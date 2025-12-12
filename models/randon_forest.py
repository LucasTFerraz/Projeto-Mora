import random

#import sklearn

n = [{"Nome":x} for x in ["Beatrice","Erika","Eva","George","Maria","Kraus","Rosa","Rudolf","Jessica","Delta"]]
c = ["Senior","Junior","Treinando","Contratado"]
n[0]["nivel"] = "Senior"
n[1]["nivel"] = "Junior"
n[2]["nivel"] = "Treinando"
n[3]["nivel"] = "Contratado"
sen = ["Beatrice"]
for x in range(4,len(n)):
    ch = random.choice(c)
    n[x]["nivel"] = ch
    if ch =="Senior":
        sen.append(n[x]["Nome"])

for x in range(len(n)):
    if n[x]["nivel"] == "Treinando":
        n[x]["Treinador"] = random.choice(sen)

print(n)
