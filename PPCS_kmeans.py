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
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from numpy import array
from array import *
import csv
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy
from itertools import *
from sklearn.utils.extmath import cartesian
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pandas.read_csv('PPCS_processed_small_streetstop kmeans.csv', skip_blank_lines=True)
runSingleVarKMeans = False
runMultVarKMeans = True
runRandomForest = True
fig1 = plt.figure(1)
fig2 = plt.figure(2)

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


#### Street Stop vars
# V34 41, 43, 45, 47, 51 - Reason for stop
def isolateYes(var_name):
  return pandas.get_dummies(df[var_name]).iloc[:][1]
  # cur_var = pandas.get_dummies(df[var_name])
  # print cur_var
  # print cur_var.iloc[:][1]
  # cur_var.columns = ['Yes', 'No', 'Refused', 'Missing']
  # # drop 'Don't know', 'Refused', and 'Missing' columns
  # cur_var.drop('No', axis=1, inplace=True)
  # # cur_var.drop('Don\'t know', axis=1, inplace=True)
  # cur_var.drop('Refused', axis=1, inplace=True)
  # cur_var.drop('Missing', axis=1, inplace=True)
  # return cur_var

v34 = isolateYes('V34') # Reason for stop was given
v39 = isolateYes('V39') # Reason for stop: Suspect you of something
v41 = isolateYes('V41') # Reason for stop: Match someone's description
v43 = isolateYes('V43') # Reason for stop: Seeking info
v45 = isolateYes('V45') # Reason for stop: Investigating crime
v47 = isolateYes('V47') # Reason for stop: Provide assistance to you
v49 = isolateYes('V49') # Reason for stop: Someone with you match description
v51 = isolateYes('V51') # Reason for stop: Someone with you suspected of something

# V66 - Officer legitimate reason for stopping
# TODO

# V67 - V73 Officer race
# def getOfficerRace(var_name):
#   return pandas.get_dummies(df[var_name])
  # cur_var = pandas.get_dummies(df[var_name])
  # cur_var.columns = ['Not selected', 'Race', 'Refused', 'Missing']
  # cur_var.drop('Not selected', axis=1, inplace=True)
  # cur_var.drop('Refused', axis=1, inplace=True)
  # cur_var.drop('Missing', axis=1, inplace=True)

v67 = isolateYes('V67') # Officer Hispanic/Latino
v68 = isolateYes('V68') # Officer Race: White
v69 = isolateYes('V69') # Officer Race: Black
# v70 = isolateYes('V70') # Officer Race: American Indian/Alaska Native
v71 = isolateYes('V71') # Officer Race: Asian
# v72 = isolateYes('V72') # Officer Race: Native Hawaiian/Pacific Islander
v73 = isolateYes('V73') # Office Race: Don't know

# V81 - Time of day
# V93 - Given written warning
# V98, 100, ..., 112 - Factors Influenced Response (to police)
# V162 - Feel police actions necessary
# V163 Feel force/threats excessive
# v348 - File complaint against police (check frequency?)


######################################################################
# HELPER FUNCTIONS

# makeNumClustersPlot plots the number of clusters against the sum of distances of points to their closest centroid
def makeNumClustersPlot(input_data, plt, fig, position, name):
  clusterRange = range(1, 20)
  kmeansRange = [KMeans(init='random', n_clusters=clusterNum, random_state=0).fit(input_data) for clusterNum in clusterRange]
  allCentroids = [k.cluster_centers_ for k in kmeansRange]
  D_k = [k.inertia_ for k in kmeansRange]
  cur_ax = fig.add_subplot(position)
  name = 'numClusters: ' + name
  cur_ax.set_xlabel(name)
  cur_ax.set_ylabel('sum of dist to closest centroid')
  cur_ax.plot(clusterRange, D_k)


# runKMeans outputs [x_final, y_final, label_freq]
def runKMeans(clusterNum, input_data, xlen, ylen, varName):
  kmeans = KMeans(init='random', n_clusters=clusterNum, random_state=0).fit(input_data)
  label_freq = [len(list(group)) for key, group in groupby(sorted(kmeans.labels_))]
  centroids = kmeans.cluster_centers_
  # print "kMeans cluster centers:\n", centroids
  print "%s: Sum of dist to closest centroid: %f" % (varName, kmeans.inertia_)

  # Separate columns associated with 'x' and 'y';
  # Choose index of column with maximum value
  x_points = centroids[:, range(xlen)]
  x_final = numpy.argmax(x_points, axis=1)
  y_points = centroids[:, range(xlen, xlen+ylen)]
  y_final = numpy.argmax(y_points, axis=1)
  return [x_final, y_final, label_freq, kmeans]

def runKMeans3(clusterNum, input_data, varName):
  kmeans = KMeans(init='k-means++', n_clusters=clusterNum, random_state=0).fit(input_data)
  # kmeans = MiniBatchKMeans(init='k-means++', n_clusters=clusterNum, random_state=0).fit(input_data)
  # kmeans = AgglomerativeClustering(n_clusters=clusterNum).fit(input_data)
  label_freq = [len(list(group)) for key, group in groupby(sorted(kmeans.labels_))]
  centroids = kmeans.cluster_centers_
  # print "kMeans cluster centers:\n", centroids
  # print "label_freq", label_freq
  print "%s: Sum of dist to closest centroid: %f" % (varName, kmeans.inertia_)
  return [kmeans, label_freq]

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

if runSingleVarKMeans:
  #### Figure 1: INCOME and RACE
  # fig1 = plt.figure(1)

  # INCOME vs. Police Behave Properly (v347)
  learn_income_v347 = pandas.concat([income, v347], axis=1)
  makeNumClustersPlot(learn_income_v347, plt, fig1, 221, 'Income')
  [x_final1, y_final1, label_freq1, kmeans1] = runKMeans(6, learn_income_v347, 3, 2, 'Income')
  ax1 = makeClusterPlot(plt, fig1, x_final1, y_final1, label_freq1, 222, 'Respondee Income (numClusters = 6)', 'Income', 'Police Behave Properly')
  ax1.set_yticklabels(['', '', 'Yes', '', '', '', '', 'No'])
  ax1.set_xticklabels(['', '0 - 20,000', '', '20,000 - 49,999', '', '$50,000 or more'])

  # RACE vs. Police Behave Properly (v347)
  learn_race_v347 = pandas.concat([race, v347], axis=1)
  makeNumClustersPlot(learn_race_v347, plt, fig1, 223, 'Race')
  [x_final2, y_final2, label_freq2, kmeans2] = runKMeans(4, learn_race_v347, 18, 2, 'Race')
  ax2 = makeClusterPlot(plt, fig1, x_final2, y_final2, label_freq2, 224, 'Respondee Race (numClusters = 4)', 'Race', 'Police Behave Properly')
  ax2.set_yticklabels(['', '', 'Yes', '', '', '', '', 'No'])
  ax2.set_xticklabels(['', 'White', '', 'Black', '', '', '', 'Asian'])

  #### Figure 2: WORK
  # fig2 = plt.figure(2)

  # # WORK vs. Police Behave Properly (v347)
  learn_work_v347 = pandas.concat([work, v347], axis=1)
  makeNumClustersPlot(learn_work_v347, plt, fig2, 421, 'Work')
  [x_final3, y_final3, label_freq3, kmeans3] = runKMeans(4, learn_work_v347, 2, 2, 'Work')
  ax3 = makeClusterPlot(plt, fig2, x_final3, y_final3, label_freq3, 422, 'Respondee Work (numClusters = 4)', 'Work', 'Police Behave Properly')
  ax3.set_yticklabels(['', '', 'Yes', '',  '', '', '', 'No'])
  ax3.set_xticklabels(['', '', 'Work last week - Yes', '', '', '', '', 'Work last week - No'])

######################################################################
# K-MEANS with multiple features (more than 2)

# TEMP COMMENT:
# for each example, get cluster label, update count of v347 value for that cluster
# after going through all examples, print most common v347 value for each cluster
# assign test examples to cluster
# find % of test examples that were placed in cluster with matching v347 value

# TEMP COMMENT:
# pred_result is cluster labels of test set
# want to check v347_yes_counts[cluster label] to see if it's a "yes"/ "no" cluster
# compare v347_yes_counts' yes/no label with the true yes/no label (found in ytest4)
def testKMeans(kmeans, X_test, Y_test, yes_counts):
  pred_result = kmeans.predict(X_test) # get k-means predicted labels
  tp = 0
  fp = 0 # cluster_label = 1; actual label = 0
  tn = 0
  fn = 0 # cluster_label = 0; actual label = 1
  for result_i in range(len(pred_result)):
    # Get the k-means predicted label for the example
    cluster_label = 1
    if yes_counts[pred_result[result_i]] == 0:
      cluster_label = 0

    # Compare the k-means predicted label with the true label
    # accuracy = tp + tn / all
    # precision = tp / tp + fp
    # recall = tp / tp + fn
    if cluster_label == 1 and Y_test.iloc[result_i]['Yes'] == 1:
      tn += 1
    if cluster_label == 1 and Y_test.iloc[result_i]['Yes'] == 0:
      fn += 1
    if cluster_label == 0 and Y_test.iloc[result_i]['Yes'] == 0:
      tp += 1
    if cluster_label == 0 and Y_test.iloc[result_i]['Yes'] == 1:
      fp += 1

  total = len(pred_result)
  print "tp, fp, tn, fn, total", tp, fp, tn, fn, total
  print "accuracy:", (1.0 * (tp + tn) / total)
  print "precision:", (1.0 * tp / (tp + fp))
  print "recall:", (1.0 * tp / (tp + fn))

if runMultVarKMeans:
  # 4: [INCOME and RACE] vs. Police Behave Properly (v347)
  income_race = pandas.concat([income, race], axis=1)
  X_train4, X_test4, y_train4, y_test4 = train_test_split(income_race, v347, random_state=0)
  makeNumClustersPlot(X_train4, plt, fig2, 423, 'Income and Race')
  [kmeans4, label_freq4] = runKMeans3(4, X_train4, 'Income and Race')
  v347_yes_counts = [0] * len(label_freq4)
  for label in kmeans4.labels_:
    v347_yes_counts[label] += y_train4.iloc[label]['Yes']
  # print "v347_yes_counts", v347_yes_counts
  for label_freq_i in range(len(label_freq4)):
    v347_yes_counts[label_freq_i] = 1.0*v347_yes_counts[label_freq_i]/(1.0 * label_freq4[label_freq_i])
  # print "v347_yes_counts", v347_yes_counts
  testKMeans(kmeans4, X_test4, y_test4, v347_yes_counts)

  # 5: [SEX, RACE, WORK, INCOME] vs. Police Behave Properly (v347)
  srwi = pandas.concat([sex, race, work, income], axis=1)
  X_train5, X_test5, y_train5, y_test5 = train_test_split(srwi, v347, random_state=0)
  makeNumClustersPlot(X_train5, plt, fig2, 424, 'Sex, Race, Work, Income')
  [kmeans5, label_freq5] = runKMeans3(10, X_train5, 'Sex, Race, Work, Income')
  v347_yes_counts_srwi = [0] * len(label_freq5)
  for label in kmeans5.labels_:
    v347_yes_counts_srwi[label] += y_train5.iloc[label]['Yes']
  for label_freq_i in range(len(label_freq5)):
    v347_yes_counts_srwi[label_freq_i] = 1.0*v347_yes_counts_srwi[label_freq_i]/(1.0 * label_freq5[label_freq_i])
  testKMeans(kmeans5, X_test5, y_test5, v347_yes_counts_srwi)

  # 6: [Reason for stop] vs. Police Behave Properly (v347)
  stop_rsn = pandas.concat([v34, v39, v41, v43, v45, v47, v49, v51], axis=1)
  X_train6, X_test6, y_train6, y_test6 = train_test_split(stop_rsn, v347, random_state=0)
  makeNumClustersPlot(X_train6, plt, fig2, 425, 'Reason for stop')
  [kmeans6, label_freq6] = runKMeans3(3, X_train6, 'Reason for stop')
  v347_yes_counts_stop_rsn = [0] * len(label_freq6)
  for label in kmeans6.labels_:
    v347_yes_counts_stop_rsn[label] += y_train6.iloc[label]['Yes']
  for label_freq_i in range(len(label_freq6)):
    v347_yes_counts_stop_rsn[label_freq_i] = 1.0*v347_yes_counts_stop_rsn[label_freq_i]/(1.0 * label_freq6[label_freq_i])
  testKMeans(kmeans6, X_test6, y_test6, v347_yes_counts_stop_rsn)

  # 7: [Officer race] vs. Police Behave Properly (v347)
  officer_race = pandas.concat([v67, v68, v69, v71, v73], axis=1)
  X_train7, X_test7, y_train7, y_test7 = train_test_split(officer_race, v347, random_state=0)
  makeNumClustersPlot(X_train7, plt, fig2, 426, 'Officer race')
  [kmeans7, label_freq7] = runKMeans3(3, X_train7, 'Officer race')
  v347_yes_counts_officer_race = [0] * len(label_freq7)
  for label in kmeans7.labels_:
    v347_yes_counts_officer_race[label] += y_train7.iloc[label]['Yes']
  for label_freq_i in range(len(label_freq7)):
    v347_yes_counts_officer_race[label_freq_i] = 1.0*v347_yes_counts_officer_race[label_freq_i]/(1.0 * label_freq7[label_freq_i])
  testKMeans(kmeans7, X_test7, y_test7, v347_yes_counts_officer_race)

  # 8: [Reason for stop, Officer race] vs. Police Behave Properly (v347)
  vars_8 = pandas.concat([v34, v39, v41, v43, v45, v47, v49, v51, v67, v68, v69, v71, v73], axis=1)
  X_train8, X_test8, y_train8, y_test8 = train_test_split(vars_8, v347, random_state=0)
  makeNumClustersPlot(X_train8, plt, fig2, 427, 'Reason for stop/Officer race')
  [kmeans8, label_freq8] = runKMeans3(4, X_train8, 'Reason for stop/Officer race')
  v347_yes_counts_8 = [0] * len(label_freq8)
  for label in kmeans8.labels_:
    v347_yes_counts_8[label] += y_train8.iloc[label]['Yes']
  for label_freq_i in range(len(label_freq8)):
    v347_yes_counts_8[label_freq_i] = 1.0*v347_yes_counts_8[label_freq_i]/(1.0 * label_freq8[label_freq_i])
  testKMeans(kmeans8, X_test8, y_test8, v347_yes_counts_8)



######################################################################
# RANDOM FOREST

if runRandomForest:
  # RF 5: [SEX, RACE, WORK, INCOME] vs. Police Behave Properly (v347)
  rf_classifier5 = RandomForestClassifier(n_estimators=5)
  rf_classifier5 = rf_classifier5.fit(X_train5, y_train5)
  rf_accuracy5 = rf_classifier5.score(X_test5, y_test5)
  print "Random Forest (Sex, Race, Work, Income) accuracy:", rf_accuracy5

  # RF 6: [Reason for stop] vs. Police Behave Properly (v347)
  rf_classifier6 = RandomForestClassifier(n_estimators=5)
  rf_classifier6 = rf_classifier6.fit(X_train6, y_train6)
  rf_accuracy6 = rf_classifier6.score(X_test6, y_test6)
  print "Random Forest (Reason for stop) accuracy:", rf_accuracy6

  # RF 7: [Officer Race] vs. Police Behave Properly (v347)
  rf_classifier7 = RandomForestClassifier(n_estimators=5)
  rf_classifier7 = rf_classifier7.fit(X_train7, y_train7)
  rf_accuracy7 = rf_classifier7.score(X_test7, y_test7)
  print "Random Forest (Officer Race) accuracy:", rf_accuracy7

  # RF 8: [Reason for stop, Officer Race] vs. Police Behave Properly (v347)
  rf_classifier8 = RandomForestClassifier(n_estimators=5)
  rf_classifier8 = rf_classifier8.fit(X_train8, y_train8)
  rf_accuracy8 = rf_classifier8.score(X_test8, y_test8)
  print "Random Forest (Reason for stop, Officer Race) accuracy:", rf_accuracy8







#### Display all Figures
plt.show()

######################################################################


