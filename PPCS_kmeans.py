#!/usr/bin/python
# Michelle Lam

import random
import collections
import math
import sys
from collections import Counter
from util import *
import openpyxl
import sklearn

def importData():
  # Insert data processing

def runKMeans():
  # Import training data as X
  clusterNum = 5
  kmeans = KMeans(n_clusters=clusterNum, random_state=0).fit(X)
  print "kMeans labels:", kmeans.labels_
  print "kMeans cluster centers:", kmeans.cluster_centers_
  print "kMeans sum of dist to closest cluster center:", kmeans.inertia_

  # Import test data as Y
  pred_result = kmeans.predict(Y)
  print "kMeans predictions:", pred_result
