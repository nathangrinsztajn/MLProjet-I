import pandas as pd

def categoricalToDummies(table):
    column_names = table.columns
    categorical_column = column_names[column_names.str[10] == 'c']
    for column in categorical_column:
        table[column] =  table[column].astype('category')
    for column in categorical_column:
        dummies = pd.get_dummies(table[column],prefix=column)
        table = pd.concat([table,dummies],axis =1)
        table.drop([column],axis=1,inplace= True)
    return table