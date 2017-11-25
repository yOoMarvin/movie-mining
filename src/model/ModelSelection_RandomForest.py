# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:51:44 2017

@author: Dan
"""

import pandas as pd
import ClassifierTemplate as ct
from time import time
from sklearn.ensemble import RandomForestClassifier
import pickle

def RandomForest(df):
    classifier = ct.Classifier(df, 'productivity_binned_binary')
    # get parameters for GridSearch
    # F1 Score with micro averaging
    score = classifier.f1(average='micro')
    # Classifier is Random Forest from sklearn
    estimator = classifier.randomForest()
    # use 10 fold cross validation as for other classifiers
    cv = classifier.fold(k=10, random_state=42)

    # contains all variables that GridSearch is comuting F1 score for
    param_grid = {"n_estimators": [10, 15, 20], # nr. of trees
                  "max_depth": [3, None], #max-Depth of trees. "None" means expansion of leaves until leaves are pure or until all leaves contain less than min_samples_split sample 
                  "max_features": [1,3,10,15,20], # nr. of features when looking for best split using measurement method defined in criterion
                  "min_samples_split": [2,3,5,10], #minimum # of samples required to split an internal node
                  "min_samples_leaf": [1,3,10], #minimum # of samples required to be at a leaf node
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]} # measuring method for best split for nodes
    
    param_grid_t = {"n_estimators": [20], # nr. of trees
                  "max_depth": [None], #max-Depth of trees. "None" means expansion of leaves until leaves are pure or until all leaves contain less than min_samples_split sample 
                  "max_features": [20], # nr. of features when looking for best split using measurement method defined in criterion
                  "min_samples_split": [10], #minimum # of samples required to split an internal node
                  "min_samples_leaf": [3], #minimum # of samples required to be at a leaf node
                  "bootstrap": [False],
                  "criterion": ["entropy"]} # measuring method for best split for nodes
    
    print(status + 'starting GridSearch')
    start = time()
    
    #store GridSearch result in variable gs
    gs = classifier.gridSearch(estimator,
                               score,
                               param_grid_t,
                               print_results=False,
                               verbose=2,
                               cv=cv)
    #Print best result of GridSearch
    classifier.gridSearchBestScore(gs)
    #save the best result
    #save_model(gs.best_params_, gs.best_score_)

    print("GridSearch for RandomForest took {} seconds".format(time()-start))
    #store to csv
    classifier.gridSearchResults2CSV(gs, param_grid, "../../data/processed/randForest2.csv")
    print(status + 'CSV with scores saved')
    return

def save_model(gs_params, gs_score):
    filename = "ModelSelection_RandomForest_BestScore"
    #pickle.dump(modelRF, open(filename, "wb"))
    name = filename+".txt"
    file = open(name,"w")
    #file.write("Best Score: "+str(modelRF))
    file.write("Best Score: "+str(gs_score)+" | "+str(gs_params))
    print(status + 'Best Score and Params saved to txt File')
    file.close() 
    return


#import data set
df = pd.read_csv("../../data/interim/only_useful_datasets.csv", index_col=0)

#some strings we need
status = '[Status: ]'

#drop not needed features
df = df.drop(["original_title",
            "adult",
            "productivity_binned_multi"],
            #"production_companies"],
            1)
RandomForest(df)

"""
#Best score up to now:
0.8248175182481752 with params
{'bootstrap': False,
'criterion': 'entropy','
max_depth': None,
'max_features': 20,
'min_samples_leaf': 3,
'min_samples_split': 10,
'n_estimators': 20}.
"""
