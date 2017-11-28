# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:44 2017

@author: Dan
"""

import pandas as pd
import ClassifierTemplate as ct
from time import time

def RandomForest(df):
    classifier = ct.Classifier(df, 'productivity_binned_binary')
    
    score = classifier.f1(average='macro')
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

    classifier.splitData()
    classifier.upsampleTrainData()

    classifier.fit_predict(gs.best_estimator_)
    print(classifier.confusion_matrix())

    classifier.classification_report()
    
    return

#import data set
df = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)

df = df.drop(["original_title",
                  "adult",
                  "productivity_binned_multi"],1)
RandomForest(df)
