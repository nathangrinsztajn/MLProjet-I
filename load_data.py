import numpy as np
import pandas as pd




#All the data is either int or float, no null value (NA is coded by -1)

def read_data(pathTrain, pathTest):

    ## par exemple pathTrain = '../train.csv' ou 'train.csv' ##
    train = pd.read_csv(pathTrain, dtype={'target': np.int8, 'id': np.int32})
    test = pd.read_csv(pathTest, dtype={'id': np.int32})

    # replace -1 by nan
    train[train == -1] = np.nan
    test[test == -1] = np.nan
    # Almost all variables have no missing data
    # Variables with missing data: ps_ind_02_cat, ps_ind_04_cat, ps_ind_05_cat,
    # ps_reg_03, ps_car_01_cat,ps_car_02_cat, ps_car_03_cat, ps_car_05_cat
    # ps_car_07_cat, ps_car_09_cat, ps_car_11,ps_car_12, ps_car_14
    # Two variables have a lot of missing: ps_car_03_cat (69%) and ps_car_05_cat (45%)
    # ps_reg_03 has 18% of missing data
    # All other variables have less that 10% of missing data
    # print(train.apply(NA_proportion,axis=0))

    # train: 595212 rows, 59 columns

    #change the type of categorical columns
    column_names = train.columns
    categorical_column = column_names[column_names.str[10] == 'c']
    for column in categorical_column:
        train[column] = train[column].astype('category')
        test[column]=test[column].astype('category')

    return(train,test)


def create_dummies(df):
    column_names = df.columns
    categorical_column = column_names[column_names.str[10] == 'c']
    for column in categorical_column:
        dummies = pd.get_dummies(df[column], prefix=column, dummy_na=True)
        df = pd.concat([df, dummies], axis=1)
        ## dropping the original columns ##
        df.drop([column], axis=1, inplace=True)
    return(df)


def replace_na(df):
    labels=list(df)
    numeric_features = [x for x in labels if x[-3:] not in ['bin', 'cat']]
    categorical_features = [x for x in labels if x[-3:] == 'cat']
    binary_features = [x for x in labels if x[-3:] == 'bin']
    df.loc[:, numeric_features]=df[numeric_features].fillna(df[numeric_features].mean())
    ## on ne remplace pas forcement les Na des categorical variables ##
    #df[categorical_features] = df[categorical_features].apply(lambda x: x.fillna(x.value_counts().index[0]))
    return (df)

def ultimeload(pathTrain, pathTest, returnID = False):
    train, test = read_data(pathTrain, pathTest)
    X_train = train.iloc[:, 2:]
    y_train = train.target
    X_test = test.iloc[:,1:]
    test_ids = test['id']
    train_ids = train['id']
    X_train, X_test = replace_na(X_train), replace_na(X_test)
    X_train, X_test = create_dummies(X_train), create_dummies(X_test)
    if returnID:
        return (y_train, X_train, X_test, test_ids, train_ids)
    else:
        return (y_train, X_train, X_test, test_ids)

def createSubmission(SubmissionName, predProb, test_ids):
    submit = pd.DataFrame({'id': test_ids, 'target': predProb})
    print(submit.head())
    submit.to_csv(SubmissionName, index=False)