# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:09:06 2017

@author: Steff
"""

import ClassifierTemplate as ct
import pandas as pd

data = {
        'col1':[0.36,0.86,0.17,0.58,0.3,0.53,0.15,0.11,0.98,0.49,0.54,0.21,0.15,0.11,0.08,0.81,0.87,0.24,0.39,0.89],
        'col2':[0.76,0.83,0.54,0.52,0.05,0.6,0.21,0.29,0.63,0.72,0.94,0.08,0.41,0.18,0.4,0.76,0.2,0.08,0.44,0.55],
        'col3':[0.48,0.22,0.51,0.45,0.29,0.49,0.61,0.82,0.81,0.28,0.63,0.1,0.53,0.54,0.03,0.14,0.06,0.76,0.38,0.15],
        'col_label':[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1]
        }

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,'col_label')

# get parameters for GridSearch
scorer = c.f1(average='micro') # use F1 score with micro averaging
estimator = c.knn() # get kNN estimator
cv = c.fold(
        k=2
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