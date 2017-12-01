# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:44 2017

@author: Dan
"""

import pandas as pd
import ClassifierTemplate as ct
from time import time

def savethresholds(df):
    classifier = ct.Classifier(df, 'productivity_binned_binary')
    print(len(classifier.data.columns))
    classifier.thresholdByColumn(3,"company")
    classifier.thresholdByColumn(8,"actor")
    classifier.thresholdByColumn(3,"director")
    print(len(classifier.data.columns))
    df.to_csv("../../data/processed/train_set_thresh.csv",encoding="utf-8")
    print("Saved succesfully")
    return

def RandomForest(df):
    classifier = ct.Classifier(df, 'productivity_binned_binary')
    #classifier.splitData()
    #classifier.upsampleTrainData()
    #classifier.downsampleTrainData()
    
    score = classifier.f1(average='macro')
    estimator = classifier.randomForest()
    cv = classifier.fold(k=10, random_state=42)
    print("Imported classifier")
    
    classifier.dropColumns([
            "original_title"
            ,"quarter"
            ,"productivity_binned_multi"
            , "adult"
    ])
    
    
    classifier.dropColumnByPrefix("belongs")
    classifier.dropColumnByPrefix("actor")
    classifier.dropColumnByPrefix("director")
    classifier.dropColumnByPrefix("production_country")
    
    print(len(classifier.data.columns))
    classifier.thresholdByColumn(3,"company")
    classifier.thresholdByColumn(8,"actor")
    classifier.thresholdByColumn(3,"director")
    print(len(classifier.data.columns))
    
    
    param_grid = {"max_depth": [3, None],
                  "max_features": [1,3,10],
                  "min_samples_split": [2,3,10],
                  "min_samples_leaf": [1,3,10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    """
    param_grid = {"max_depth": [None],
                  "max_features": [5],
                  "min_samples_split": [2],
                  "min_samples_leaf": [1],
                  "bootstrap": [True],
                  "criterion": ["entropy"]}
    """
    start = time()

    print("Starting Grid Search")
    
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
    
    
    """OLD
    # calculate cross validation: try samplings
    estimator.set_params(
        max_depth=None,
        max_features=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        criterion="entropy"
    )"""
        # calculate cross validation: try samplings
    estimator.set_params(
        max_depth=None,
        max_features=5,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        criterion="entropy"
    )
    print(classifier.cross_validate(cv,estimator,sample=""))

    """
    #run classifier on test set and print classification report
    classifier.fit_predict(gs.best_estimator_)
    print(classifier.confusion_matrix())
    classifier.classification_report()
    """ 
    return

#import data set
df = pd.read_csv("../../data/processed/train_set.csv", index_col=0)
print("Import done")
#savethresholds(df)
RandomForest(df)
#print(df.shape)

"""
--------------------------- GRID SEARCH BEST SCORE ---------------------------
02.12.2017: Best score is 0.5824105943305237 with params {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}.
            {'f1': 0.57256622981701921, 'no': 640, 'yes': 3273} //collection, actor, director, prod_comp, quarter dropped
------------------------------------------------------------------------------
"""

"""
--------------------------- GRID SEARCH ALL SCORES ---------------------------
01.12.2017: Best score is 0.5628858470360576 with params {'bootstrap': True, 'criterion': 'entropy', 'max_depth': None, 'max_features': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}.
            {'f1': 0.52541654006451621, 'no': 411, 'yes': 3502} //non dropped
02.12.2017: Best score is 0.5496065013234092 with params {'bootstrap': True, 'criterion': 'entropy', 'max_depth': None, 'max_features': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}.
            {'f1': 0.54776085832391819, 'no': 523, 'yes': 3390} with threshold, dropped collection
02.12.2017: Best score is 0.5812527458428057 with params {'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}.            
            {'f1': 0.56272840734489338, 'no': 589, 'yes': 3324} //collection, actor, director, prod_comp
02.12.2017: Best score is 0.5824105943305237 with params {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}.
            {'f1': 0.57256622981701921, 'no': 640, 'yes': 3273} //collection, actor, director, prod_comp
------------------------------------------------------------------------------
"""
