import json 
import numpy
import pandas
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
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

print "Filtered shape of CSV"
print df.shape

features = df.columns.values

# Classification vector: "POLICE BEHAVE PROPERLY - V347"
# Look into multi-class classification.
y = sparse['V347']
y[y == 2] = 0  # Police did not behave properly 
y[y == 8] = 0
y[y == 3] = 0
y[y == 1] = 1  # Police behaved properly

# Preprocess the data for modeling.
sparse.drop('V347', axis=1, inplace=True)

# Train with logistic regression model.
X_train, X_test, y_train, y_test = train_test_split(sparse, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print "The accuracy from training on Logistic Regression is: "
print(metrics.accuracy_score(y_test, y_pred_class))
confusion_matrix(y_test, y_pred_class)
confusion_matrix = pandas.crosstab(y_test, y_pred_class, rownames=['Actual'], colnames=['Predicted'], margins=True)
print confusion_matrix

# Baseline: null accuracy -- the accuracy that can be achieved by predicting most frequent class.
zeros = 1 - y_test.mean()  # The percentage of 0's (behaved improperly)
print "The baseline null accuracy is: "
print max(y_test.mean(), 1 - y_test.mean())

# Train with KNN model. 
# Tune the parameters of K, as well as the weight of features (weighted as 1/d depending on distance of a point from the cluster mean)
knn = KNeighborsClassifier()
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']

param_grid = dict(n_neighbors = k_range, weights = weight_options)
grid = GridSearchCV(knn, param_grid, cv=5, scoring = 'accuracy')
grid.fit(sparse, y)

print "The best score and best params are: "
print(grid.best_score_)
print(grid.best_params_)