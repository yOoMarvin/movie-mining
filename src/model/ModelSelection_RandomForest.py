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
    #classifier.splitData()
    #classifier.upsampleTrainData()
    #classifier.downsampleTrainData()
    
    score = classifier.f1(average='macro')
    estimator = classifier.randomForest()
    cv = classifier.fold(k=10, random_state=42)
    
    classifier.dropColumns([
            "original_title"
            ,"quarter"
            ,"productivity_binned_multi"
    ])
    
    classifier.dropColumnByPrefix("actor")
    classifier.dropColumnByPrefix("quarter")

    
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
                               cv=cv
                               #,onTrainSet=True
                               )
    classifier.gridSearchBestScore(gs)
    print("GridSearch for RandomForest took {}".format(time()-start))
    classifier.gridSearchResults2CSV(gs, param_grid, "randForest.csv")

    
    
    # calculate cross validation: try samplings
    estimator.set_params(
        max_depth=None,
        max_features=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        criterion="entropy"
    )
    print(classifier.cross_validate(cv,estimator,sample=""))


    #run classifier on test set and print classification report
    #classifier.fit_predict(gs.best_estimator_)
    #print(classifier.confusion_matrix())
    #classifier.classification_report()
    
    return

#import data set
df = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)

RandomForest(df)

"""
--------------------------- GRID SEARCH BEST SCORE ---------------------------
  Best score is 0.580698370927875 with params {'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}.
------------------------------------------------------------------------------
"""