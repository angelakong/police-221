#!/usr/bin/python
# Catherine Xu

import random
import collections
import math
import sys
import pandas
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import *
import numpy
from numpy import array

df_allegations = pandas.read_csv('CPDB data/Allegations.csv', skip_blank_lines=True)

allegation_count_colnum = 12

# Load Data
# ---------------------------------------------------------------------------

# Allegation location
location = pandas.get_dummies(df_allegations['location'])
location.columns = [
                    '01',
                    '014',
                    '02',
                    '03',
                    '04',
                    '05',
                    '06',
                    '07',
                    '08',
                    '09',
                    '10',
                    '11',
                    '12',
                    '13',
                    '14',
                    '15',
                    '16',
                    '17',
                    '18',
                    '19',
                    '20',
                    'XX']
location.drop('014', axis=1, inplace=True)
location.drop('XX', axis=1, inplace=True)

# Allegation result
result = pandas.get_dummies(df_allegations['result'])
result.columns = ['Sustained', 'Unknown', 'Unsustained']
result.drop('Unknown', axis=1, inplace=True)

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

# makeClusterPlot plots all clusters, with the size of the point proportional in size to the number of examples in that cluster
def makeClusterPlot(plt, fig, x_final, y_final, label_freq, position, title, xlabel, ylabel):
  cur_ax = fig.add_subplot(position)
  # cur_ax.set_title(title)
  cur_ax.set_xlabel(xlabel)
  cur_ax.set_ylabel(ylabel)
  plt.scatter(x_final, y_final, s=label_freq)
  return cur_ax

# Generate k-means plot for Allegation location vs. Allegation result
# ---------------------------------------------------------------------------
fig1 = plt.figure(1)

location_result = pandas.concat([location, result], axis=1)
makeNumClustersPlot(location_result, plt, fig1, 221, 'Allegation location')
[x_final1, y_final1, label_freq1, kmeans1] = runKMeans(6, location_result, 20, 2, 'Income')
ax1 = makeClusterPlot(plt, fig1, x_final1, y_final1, label_freq1, 222, 'Allegation location', 'Location', 'Complained sustained status')
# ax1.set_yticklabels(['', '', 'Yes', '', '', '', '', 'No'])
# ax1.set_xticklabels(['', '0 - 20,000', '', '20,000 - 49,999', '', '$50,000 or more'])

# Display all Figures
plt.show()







