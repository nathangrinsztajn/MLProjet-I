import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import time
import gc
import load_data as ld
from Gini import *

(y, X, X_test, test_ids) = ld.ultimeload('train.csv', 'test.csv')

y_valid_pred = 0*y
y_test_pred = 0

MAX_ROUNDS = 5000
OPTIMIZE_ROUNDS = True
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 40 #50

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)




# Run CV

for i, (train_index, test_index) in enumerate(kf.split(X)):

    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    #X_test = test_df.copy()
    print( "\nFold ", i)

    # setup model
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

    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        eval_set = [(X_valid, y_valid)]
        fit_model = model.fit(X_train, y_train,
                              eval_set=eval_set,
                              eval_metric=gini_xgb,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS
                              )
        print("  Best N trees = ", model.best_ntree_limit)
        print("  Best gini = ", model.best_score)
    else:
        fit_model = model.fit(X_train, y_train)

    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid, ntree_limit= model.best_ntree_limit)[:, 1]
    print("  Gini = ", gini_normalizedc(y_valid, pred))
    y_valid_pred.iloc[test_index] = pred

    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test, ntree_limit= model.best_ntree_limit)[:, 1]

    del y_valid, X_train, X_valid, y_train


y_test_pred /= K  # Average test set predictions

print("\nGini for full training set:")
print(gini_normalizedc(y, y_valid_pred))

ld.createSubmission("nouveauXGB3.csv", y_test_pred, test_ids)
