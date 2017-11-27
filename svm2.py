# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:50:54 2017

@author: fwn
"""

from load_data import *
from sklearn import svm

y_train, X_train, X_test, test_ids = ultimeload("../train.csv", "../test.csv")

clf = svm.SVC(verbose=True)
clf.fit(X_train, y_train)   
y_test=clf.predict(X_test)

createSubmission("SubmissionName.csv", y_test, test_ids)