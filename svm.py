import json 
import numpy
import pandas
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from numpy import array
from array import *
import csv
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy.stats import mode
import matplotlib.pyplot as plt


df = pandas.read_csv('PPCS_processed_small_streetstop.csv', skip_blank_lines=True)
sparse = df.copy(deep=True)

print "Original shape of csv"
print df.shape

# Filter all the rows where the police behave properly is not equal to -9.
df.loc[df['V346'] != -9]

print "Filtered shape of CSV"
print df.shape

features = df.columns.values

# Classification vector: "POLICE BEHAVE PROPERLY - V347"
y = df['V346']

# Preprocess the data for modeling.
for f in features:
	delete = ['V346']
	if any(word in f for word in delete):
		sparse.drop(f, axis=1, inplace=True)

x_learn = sparse[1:500]
y_learn = y[1:500]
print x_learn.shape
print y_learn.shape

clf = svm.SVC()
clf.fit(x_learn,y_learn)

numTestData = 500
x_test = sparse[500:1000]
y_test = y[500:1000]

# SVM
# predictions = []
# predictions = clf.predict(x_test)

# error = sum(y_test * predictions <= 0) / numTestData
# print("Test error for SVM: " + str(error))

# Linear Regression

# fit OLS regression 
est = LinearRegression(fit_intercept=True, normalize=True)
est.fit(x_train, y_train)

# Test data that was not used for fiting

