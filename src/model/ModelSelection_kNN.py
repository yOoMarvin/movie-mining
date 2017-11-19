# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:09:06 2017

@author: Steff
"""

import ClassifierTemplate as ct
import pandas as pd

data = pd.read_csv("../../data/processed/train_set.csv", index_col=0)

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned")

# drop columns not needed for Classification
c.dropColumns([
        "original_title",
        "id.1"
])

# get parameters for GridSearch
scorer = c.f1(average="micro") # use F1 score with micro averaging
estimator = c.knn() # get kNN estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    "n_neighbors":range(3,10)
    ,"algorithm":[
            "auto"
            ,"ball_tree"
            ,"kd_tree"
            ,"brute"
    ]
    ,"weights":[
            "uniform"
            ,"distance"
    ]
    ,"p":[
            1
            ,2
            ,3
    ]
    ,"metric":[
            "euclidean"
            ,"manhattan"
            ,"chebyshev"
            ,"minkowski"
            #,"wminkowski" # throws error: additional metric parameters might be missing
            #,"seuclidean" # throws error: additional metric parameters might be missing
            #,"mahalanobis" # throws error: additional metric parameters might be missing
    ]
    # parameter can be used to tweak parallel computation / n = # of jobs
    #,"n_jobs":[1]
}


# compute GridSearch
gs = c.gridSearch(
        estimator
        ,scorer
        ,parameters
        ,print_results=False # let verbose print the results
        ,verbose=2
        ,cv=cv
)

# print best result
c.gridSearchBestScore(gs)

# save all results in csv
c.gridSearchResults2CSV(gs,parameters,"kNN_results.csv")