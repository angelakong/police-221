# -*- coding: utf-8 -*-
"""
Created on Tue May 17 02:19:31 2016
@author: angelakong
"""

import json
import numpy
import pandas
from sklearn import svm
from numpy import array
from array import *
import csv
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy.stats import mode
import matplotlib.pyplot as plt

df = pandas.read_csv('yelp_academic_dataset_business.csv', skip_blank_lines=True)
# There are 77445 training examples, 98 possible features
sparse = df.copy(deep=True)

features = df.columns.values

#for index, feature in enumerate(features):
#    if "city" in feature:
#        print index
#        
category_index = 20
city_index = 58

# Classification vector
y = df['stars']
y[y < 4.0] = 0
y[y >= 4.0] = 1


# Only consider the training data if in a certain city, and is of category: restaurant
#for i, row in sparse.iterrows():
#    if "u'Restaurants" not in row[category_index]:
#        sparse.drop(sparse.index[i])
#    if "Pittsburgh" not in row[city_index]:
#        sparse.drop(sparse.index[i])

# Preprocess the data for modeling
for f in features:
    # Special code for categories
    delete = ["hours", "city", "state", "business_id", "stars", "type", "categories", "name", "full_address", "BYOB", "neighborhoods"]
    
    if any(word in f for word in delete):
        sparse.drop(f, axis=1, inplace=True)
    else:
        
        if "Smoking" in f:
            sparse[f] = sparse[f].replace(['no', 'outdoor', 'yes'], [0,1,2])
            
        if "Attire" in f:
            sparse[f] = sparse[f].replace(['casual', 'dressy', 'formal'], [0,1,2])
        
        if "Noise" in f:
            sparse[f] = sparse[f].replace(['quiet', 'average', 'loud', 'very_loud'], [0,1,2,3])
        
        if "Wi-Fi" in f:
            sparse[f] = sparse[f].replace(['no', 'paid', 'free'], [0,1,2]) 

        if "Alcohol" in f:
            sparse[f] = sparse[f].replace(['none', 'beer_and_wine', 'full_bar'], [0,1,2])

        sparse[f].fillna(0, inplace = True)
        if "FALSE" in sparse[f].tolist(): #Convert true/false to integer values
            sparse[f] = sparse[f].astype(int)
        

#alc = df["attributes.Alcohol"].replace(['none', 'beer_and_wine', 'full_bar'], [0,1,2])
#alc.fillna(mode(alc).mode[0], inplace = True)
#
#takeout = df["attributes.Take-out"]
#takeout.fillna(0, inplace = True)
#takeout = takeout.astype(int)
#
#waiter = df["attributes.Waiter Service"]
## If restaurant does not list a benefit, we assume that it lacks it
#waiter.fillna(0, inplace = True)
#
## Fill the empty cells with the mode of category (one that appears the most)
##waiter.fillna(mode(waiter).mode[0], inplace = True)
#waiter = waiter.astype(int)
#
#casual_ambience = df["attributes.Ambience.casual"]
#casual_ambience.fillna(0, inplace = True)
##casual_ambience.fillna(mode(casual_ambience).mode[0], inplace = True)
#casual_ambience = casual_ambience.astype(int)

X = sparse[1:500]
ytrain = y[1:500]
print X.shape
print ytrain.shape

#for kernel in ('linear', 'poly', 'rbf'):

clf = svm.SVC()
clf.fit(X,ytrain)

numTestData = 1000
ytest = y[1000:2000]

predictions = []
predictions = clf.predict(sparse[1000:2000])

error = sum(ytest * predictions <= 0) / numTestData
print("Test error for SVM: " + str(error))