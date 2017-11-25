import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from load_data import *
from gini_score import *

train_data,test_data=read_data('../train.csv', '../test.csv')
train_data=replace_na(train_data)
test_data=replace_na(test_data)

#select most interestning variables (derived from our analysis)
# train_data=train_data.loc[:,["target","ps_ind_01","ps_ind_02_cat","ps_ind_03",
#                               "ps_ind_05_cat","ps_ind_06_bin","ps_ind_07_bin",
#                               "ps_ind_08_bin","ps_ind_09_bin","ps_ind_12_bin",
#                               "ps_ind_15","ps_ind_16_bin","ps_ind_17_bin","ps_car_02_cat",                              "ps_car_03_cat","ps_car_07_cat","ps_car_08_cat","ps_car_11",
#                               "ps_car_12","ps_car_13","ps_car_14","ps_reg_01",
#                               "ps_reg_02","ps_reg_03"]]

#we find that selecting a subset of variables ends up doing worse than takin all variables
#when performing randomForest


train_data=create_dummies(train_data)
test_data=create_dummies(test_data)

#we skip index and target columns
X=train_data.iloc[:,2:]
y= train_data.target


RF_model_cat= RandomForestClassifier(300,
                                     n_jobs = -1, min_samples_leaf = 50)


RF_model_cat.fit(X, y)
y_pred_RF_prob = RF_model_cat.predict_proba(test_data.iloc[:,1:])


#To get importance of variables, uncomment:
#indices=sorted(range(len(RF_model_cat.feature_importances_)), key=lambda k: RF_model_cat.feature_importances_[k])
#importance=[(RF_model_cat.feature_importances_[i],X.axes[1][i]) for i in indices]
#print(importance)


#we pre-process the result to perform gini test
y_pred_prob=[y_pred_RF_prob[i][1] for i in range(len(y_pred_RF_prob))]

submit = pd.DataFrame({'id':test_data['id'],'target':y_pred_prob})
print(submit.head())
submit.to_csv('out.csv',index=False)


