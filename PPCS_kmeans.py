#!/usr/bin/python
# Michelle Lam
# November 2016
# PPCS_kmeans.py

import json
import numpy
import pandas
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from numpy import array
from array import *
import csv
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy
from itertools import groupby

df = pandas.read_csv('PPCS_processed_small_streetstop kmeans.csv', skip_blank_lines=True)

######################################################################
# LOAD DATA BROKEN DOWN CATEGORICALLY

# SEX
sex = pandas.get_dummies(df['SEX'])
sex.columns = ['Male', 'Female']

# HISP
hisp = pandas.get_dummies(df['HISP'])
hisp.columns = ['Hispanic', 'Not Hispanic', 'Residue']

# RACE
race = pandas.get_dummies(df['RACE'])
# note: value 13 and 17 do not occur, so their labels have been omitted
race.columns = ['White Only',
                'Black Only',
                'American Indian, Alaskan Native Only',
                'Asian Only',
                'Hawaiian/Pacific Islander Only',
                'White-Black',
                'White-American Indian',
                'White-Asian',
                'White-Hawaiian',
                'Black-American Indian',
                'Black-Asian',
                'Black-Hawaiian/Pacific Islander',
                'Asian-Hawaiian/Pacific Islander',
                'White-Black-American Indian',
                'White-Black-Asian',
                'White-Asian-Hawaiian',
                '2 or 3 Races',
                '4 or 5 Races']

# WORK
work = pandas.get_dummies(df['WORK'])
work.columns = ['Work last week - Yes', 'Work last week - No', 'Residue']
work.drop('Residue', axis=1, inplace=True)

# INCOME
income = pandas.get_dummies(df['INCOME'])
income.columns = ['Less than $20,000 or N/A', '$20,000 - $49,999', '$50,000 or more']

# V347 - Police Behave Properly
v347 = pandas.get_dummies(df['V347'])
v347.columns = ['Yes', 'No', 'Don\'t know', 'Refused']
# drop 'Don't know' and 'Refused' columns
v347.drop('Don\'t know', axis=1, inplace=True)
v347.drop('Refused', axis=1, inplace=True)


######################################################################
# HELPER FUNCTIONS

# makeNumClustersPlot plots the number of clusters against the sum of distances of points to their closest centroid
def makeNumClustersPlot(input_data, plt, fig, position):
  clusterRange = range(1, 20)
  kmeansRange = [KMeans(init='random', n_clusters=clusterNum, random_state=0).fit(input_data) for clusterNum in clusterRange]
  allCentroids = [k.cluster_centers_ for k in kmeansRange]
  D_k = [k.inertia_ for k in kmeansRange]
  cur_ax = fig.add_subplot(position)
  cur_ax.set_xlabel('numClusters')
  cur_ax.set_ylabel('sum of dist to closest centroid')
  cur_ax.plot(clusterRange, D_k)


# runKMeans outputs [x_final, y_final, label_freq]
def runKMeans(clusterNum, input_data, xlen, ylen, varName):
  kmeans = KMeans(init='random', n_clusters=clusterNum, random_state=0).fit(input_data)
  label_freq = [len(list(group)) for key, group in groupby(sorted(kmeans.labels_))]
  centroids = kmeans.cluster_centers_
  # print "kMeans cluster centers:\n", centroids
  print "%s: Sum of dist to closest centroid: %f" % (varName, kmeans.inertia_)

  # TODO: adjust number of x_points and y_points
  # Separate columns associated with 'x' and 'y';
  # Choose index of column with maximum value
  x_points = centroids[:, range(xlen)]
  x_final = numpy.argmax(x_points, axis=1)
  y_points = centroids[:, range(xlen, xlen+ylen)]
  y_final = numpy.argmax(y_points, axis=1)
  return [x_final, y_final, label_freq]


# makeClusterPlot plots all clusters, with the size of the point proportional in size to the number of examples in that cluster
def makeClusterPlot(plt, fig, x_final, y_final, label_freq, position, title, xlabel, ylabel):
  cur_ax = fig.add_subplot(position)
  # cur_ax.set_title(title)
  cur_ax.set_xlabel(xlabel)
  cur_ax.set_ylabel(ylabel)
  plt.scatter(x_final, y_final, s=label_freq)
  return cur_ax

######################################################################
# RUN K-MEANS ON VARIOUS SETS OF FEATURES

#### Figure 1: INCOME and RACE
fig1 = plt.figure(1)

# INCOME vs. Police Behave Properly (v347)
learn_income_v347 = pandas.concat([income, v347], axis=1)
makeNumClustersPlot(learn_income_v347, plt, fig1, 221)
[x_final1, y_final1, label_freq1] = runKMeans(6, learn_income_v347, 3, 2, 'Income')
ax1 = makeClusterPlot(plt, fig1, x_final1, y_final1, label_freq1, 222, 'Respondee Income (numClusters = 6)', 'Income', 'Police Behave Properly')
ax1.set_yticklabels(['', '', 'Yes', '', '', '', '', 'No'])
ax1.set_xticklabels(['', '0 - 20,000', '', '20,000 - 49,999', '', '$50,000 or more'])

# RACE vs. Police Behave Properly (v347)
learn_race_v347 = pandas.concat([race, v347], axis=1)
makeNumClustersPlot(learn_race_v347, plt, fig1, 223)
[x_final2, y_final2, label_freq2] = runKMeans(4, learn_race_v347, 18, 2, 'Race')
ax2 = makeClusterPlot(plt, fig1, x_final2, y_final2, label_freq2, 224, 'Respondee Race (numClusters = 4)', 'Race', 'Police Behave Properly')
ax2.set_yticklabels(['', '', 'Yes', '', '', '', '', 'No'])
ax2.set_xticklabels(['', 'White', '', 'Black', '', '', '', 'Asian'])

#### Figure 2: WORK and __
fig2 = plt.figure(2)

# WORK vs. Police Behave Properly (v347)
learn_work_v347 = pandas.concat([work, v347], axis=1)
makeNumClustersPlot(learn_work_v347, plt, fig2, 221)
[x_final3, y_final3, label_freq3] = runKMeans(4, learn_work_v347, 2, 2, 'Work')
ax3 = makeClusterPlot(plt, fig2, x_final3, y_final3, label_freq3, 222, 'Respondee Work (numClusters = 4)', 'Work', 'Police Behave Properly')
ax3.set_yticklabels(['', '', 'Yes', '',  '', '', '', 'No'])
ax3.set_xticklabels(['', '', 'Work last week - Yes', '', '', '', '', 'Work last week - No'])



#### Display all Figures
plt.tight_layout(pad=1)
plt.show()

######################################################################
# CLASSIFY NEW EXAMPLES INTO CLUSTERS

# TODO



# # Import test data as Y
# pred_result = kmeans.predict(Y)
# print "kMeans predictions:", pred_result

# numTestData = 500
# x_test = sparse[500:1000]
# y_test = y[500:1000]



