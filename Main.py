from Gini import *
import numpy as np
from data import *
import pandas as pd
from GestionNA import *
from PreProcess import *
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout


train = dropNA(train)
print(train.shape)

#drop id et target
X = train.iloc[:,2:]
y= train.target

X = categoricalToDummies(X)
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Xshape = " + str(X_train.shape[1]))
y_train = y_train.as_matrix()
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

# create model
model = Sequential()
model.add(Dense(200, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, nb_epoch=400, batch_size=32,  verbose=2)

yPred = model.predict(X_test)
print("yPred = ")
print(yPred)

# round predictions
yPredClass = [int(round(x[0])) for x in yPred]
print("yPredClass = ")
print(sum(yPredClass))
print(len(yPred), len(y_test))
print(gini_normalized(y_test, yPred))
print(pd.crosstab(np.array(yPredClass), np.array(y_test)))