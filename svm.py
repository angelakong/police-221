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

# Filter all the rows where the police behave properly or did not behave properly.
df.loc[df['V347'] != 9]

print "Filtered shape of CSV"
print df.shape

features = df.columns.values

# Classification vector: "POLICE BEHAVE PROPERLY - V347"
y = sparse['V347']
y[y == 2] = 0  # Police did not behave properly 
y[y == 1] = 1  # Police behaved properly

# Preprocess the data for modeling.
sparse.drop('V347', axis=1, inplace=True)

x_learn = sparse[0:500]
y_learn = y[0:500]

clf = svm.SVC()
clf.fit(x_learn,y_learn)

numTestData = 500
x_test = sparse[500:1000]
y_test = y[500:1000]
numTestData = 500

# Create a linear regression object, train using the training sets
regr = LinearRegression(fit_intercept=True, normalize=True)
regr.fit(x_learn, y_learn)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % numpy.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))

# Plot outputs
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, regr.predict(x_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()