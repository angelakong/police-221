import json 
import numpy as np
import pandas
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

df = pandas.read_csv('PPCS_processed_small_streetstop.csv', skip_blank_lines=True)
X = df.copy(deep=True)

names = list(df.columns.values)
print "names"
print names

# # Baseline: null accuracy -- the accuracy that can be achieved by predicting most frequent class.
# ones = y_test.mean()  # The percentage of 1's (behaved improperly)
# print ones
# print "The baseline null accuracy is: " + str(max(ones, 1 - ones))
y = X['V347']
y[y == 1] = 0  # Police behaved properly
y[y == 8] = 0
y[y == 3] = 0
y[y == 2] = 1  # Police did not behave properly 

# Preprocess the data for modeling.
X.drop('V347', axis=1, inplace=True)

# Train with linear model
# Using cross-validation to prevent test set from "leaking" into the model such that the evaluation metrics
# do not generalize well.
# print "Training with linear model"
# clf = svm.SVC(kernel = "linear")
# scores = cross_validation.cross_val_score(clf, X, y, cv=5)
# print scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Train with logistic regression model.

# Feature selection
print X.shape
logreg = LogisticRegression()
print "rfe"
rfe = RFE(logreg, 20)
rfe = rfe.fit(X, y)
X = rfe.transform(X)
print X.shape
print (rfe.ranking_)

# Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# logreg.fit(X, y)
# y_pred_class = logreg.predict(X_test)
# print "The accuracy from training on Logistic Regression is: "
# print(metrics.accuracy_score(y_test, y_pred_class))

# # Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores = ['recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(metrics.classification_report(y_true, y_pred))
#     print()

# # Baseline: null accuracy -- the accuracy that can be achieved by predicting most frequent class.
# zeros = y_test.mean()  # The percentage of 1's (behaved improperly)
# print "The baseline null accuracy is: "
# print max(zeros, 1 - zeros)

# Train with RandomForestClassifier to reduce the imbalance of data problem.
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=10196,
                           n_features=20,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), names), 
             reverse=True)
y_true, y_pred = y_test, forest.predict(X_test)
print(metrics.classification_report(y_true, y_pred))


importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Train with KNN model. 
# Tune the parameters of K, as well as the weight of features (weighted as 1/d depending on distance of a point from the cluster mean)
knn = KNeighborsClassifier()
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']

param_grid = dict(n_neighbors = k_range, weights = weight_options)
grid = GridSearchCV(knn, param_grid, cv=5, scoring = 'accuracy')
grid.fit(X, y)

print "The best score and best params are: "
print(grid.best_score_)
print(grid.best_params_)