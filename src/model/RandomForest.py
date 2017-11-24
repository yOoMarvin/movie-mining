#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:44 2017

@author: Dan
"""

import pandas as pd
#import numpy as np
import ClassifierTemplate as ct
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
from time import time


#drop unneeded columns
def dropcolumns(df, columnName):
    df = df.drop([columnName],1)
    return df

def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def RandomForestOne(df):

    df['productivity_binned'] = pd.factorize(df['productivity_binned'])[0]
    y = pd.factorize(df['productivity_binned'])[0]

    # Create a list of the feature column's names
    cols = df.columns.tolist()

    cols.insert(0, cols.pop(cols.index('productivity_binned')))
    #reordering df using function df.reindex() 
    df = df.reindex(columns= cols)

    features = df.columns[:]
      

    #convert string information (yes/no) to float numbers
    df['productivity_binned'] = pd.factorize(df['productivity_binned'])[0]
    y = pd.factorize(df['productivity_binned'])[0]
    #print(y)

    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(df, y)

    #Classifier was now learned
    #Now: Test!
    
    #import test set
    test = pd.read_csv("../../data/processed/test_set.csv", index_col=0)
    test = dropcolumns(test, 'original_title')

    #convert string information (yes/no) to float numbers
    test['productivity_binned'] = pd.factorize(test['productivity_binned'])[0]
    features = test.columns[0:]

    #do the prediction!
    preds = clf.predict(test[features])
    #give probability of how sure classifier is about preds
    print(clf.predict_proba(test[features])[0:10])

    #create confusion matrix
    cf = pd.crosstab(test['productivity_binned'], preds, rownames=['Actual prod'], colnames=['Predicted prod'])
    print(cf)


def RandomForestTwo(df):
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('productivity_binned')))
    #reordering df using function df.reindex() 
    df = df.reindex(columns=cols)
    #convert yes/no to 0/1
    df['productivity_binned'] = pd.factorize(df['productivity_binned'])[0]
    
    train_x, test_x, train_y, test_y = train_test_split(df, [0.7,0.3], df.columns[1:], df.columns[0])
    train_test_split()
    # Train and Test dataset size details
    print ("Train_x Shape :: ", train_x.shape)
    print ("Train_y Shape :: ", train_y.shape)
    print ("Test_x Shape :: ", test_x.shape)
    print ("Test_y Shape :: ", test_y.shape)
    #trained_model = random_forest_classifier(train_x, train_y)
    #print ("Trained model :: ", trained_model)
    return

def RandomForestThree(df):
    classifier = ct.Classifier(df, 'productivity_binned')
    print(list(df))
    #dropcolumns(df, "original_title")
    score = classifier.f1(average='micro')
    estimator = classifier.randomForest()
    cv = classifier.fold(k=10, random_state=42)

    
    param_grid = {"max_depth": [3, None],
                  "max_features": [1,3,10],
                  "min_samples_split": [2,3,10],
                  "min_samples_leaf": [1,3,10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    start = time()
    gs = classifier.gridSearch(estimator,
                               score,
                               param_grid,
                               print_results=False,
                               verbose=2,
                               cv=cv)
    classifier.gridSearchBestScore(gs)
    print("GridSearch for RandomForest took {}".format(time()-start))
    classifier.gridSearchResults2CSV(gs, param_grid, "randForest.csv")
    return

#import train set
df = pd.read_csv("../../data/processed/train_set.csv", index_col=0)
df = dropcolumns(df, 'original_title')

RandomForestThree(df)

"""
RandomForestClassifier
(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2,
 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’,
 max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
 bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
 warm_start=False, class_weight=None)
"""

