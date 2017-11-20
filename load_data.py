import numpy as np
import pandas as pd




#All the data is either int or float, no null value (NA is coded by -1)

def read_data():
    train = pd.read_csv('../train.csv',index_col=0)
    test = pd.read_csv('../test.csv',index_col=0)

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

    return(train,test)

def replace_na(df):
    labels=list(df)
    numeric_features = [x for x in labels if x[-3:] not in ['bin', 'cat']]
    categorical_features = [x for x in labels if x[-3:] == 'cat']
    binary_features = [x for x in labels if x[-3:] == 'bin']

    df[numeric_features]=df[numeric_features].fillna(df[numeric_features].mean(),inplace=True)
    df[categorical_features] = df[categorical_features].apply(lambda x: x.fillna(x.value_counts().index[0]))
    return (df)



