#import pandas as pd
import numpy as np
import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from load_data import *
from gini_score import *

MAX_ROUNDS = 400
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50  

train_data,test_data=read_data()
train_data=replace_na(train_data)
test_data=replace_na(test_data)

train_data=create_dummies(train_data)
test_data=create_dummies(test_data)

#we skip index and target columns
X=train_data.iloc[:,2:]
y= train_data.target

#we split our train set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

#after trying several parameters, it turns out those paramater are near optimal
model = XGBClassifier(    
                        n_estimators=MAX_ROUNDS,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=LEARNING_RATE, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                     )


model.fit(X_train, y_train)
y_pred_RF_prob = model.predict_proba(X_test)
y_pred_RF_class = model.predict(X_test)



#we pre-process the result to perform gini test
y_pred_prob=[y_pred_RF_prob[i][1] for i in range(len(y_pred_RF_prob))]


print(gini_normalized(y_test,y_pred_prob))
