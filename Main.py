from Gini import *
import numpy as np
from data import *
import pandas as pd
from GestionNA import *
from PreProcess import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier

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

def baseline_model():
    model = Sequential()
    model.add(
        Dense(
            200,
            input_dim=X_train.shape[1],
            kernel_initializer='glorot_normal',
        ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(50, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(25, kernel_initializer='glorot_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
    return model

model = Sequential()
model.add(Dense(1000, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[gini_normalizedc])


# Fit the model
model.fit(X_train, y_train, nb_epoch=700, batch_size=32,  verbose=2)

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