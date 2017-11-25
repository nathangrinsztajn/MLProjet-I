from load_data import *

(train,test) = read_data('train.csv', 'test.csv')
train_data=replace_na(train)
test_data=replace_na(test)

train_data=create_dummies(train_data)
test_data=create_dummies(test_data)

#we skip index and target columns
X=train_data.iloc[:,2:]
y= train_data.target