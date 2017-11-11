import pandas as pd

train = pd.read_csv("train.csv", na_values="-1")
test = pd.read_csv("test.csv", na_values="-1")

#categories : ind, reg, car, calc, target, id
#type : bin, cat, qua (quantitative)

print (train.info())

def getCategorie(name):
    s = name.split("_")
    if len(s)==1:
        return name
    else:
        return s[1]

def getType(name):
    s = name.split("_")
    if len(s)==4:
        return s[3]
    else:
        return "qua"

