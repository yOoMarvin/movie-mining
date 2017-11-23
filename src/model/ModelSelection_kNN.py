# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:09:06 2017

@author: Steff
"""

import ClassifierTemplate as ct
import pandas as pd

data_train = pd.read_csv("../../data/processed/train_set.csv", index_col=0)
data_test = pd.read_csv("../../data/processed/test_set.csv", index_col=0)
data = pd.concat([data_train,data_test])

# DataFrame containing label (!)
df = pd.DataFrame(data)

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,"productivity_binned")

# drop columns not needed for Classification
c.dropColumns([
        "original_title"
        ,"budget"
        ,"adult"
])

c.dropColumnByPrefix("actor") # without 30%, with 32%
c.dropColumnByPrefix("director")
c.dropColumnByPrefix("company")
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
c.dropColumnByPrefix("quarter")
c.dropColumnByPrefix("Unnamed")

df = c.data.loc[19898]
#df = df.iloc[df.nonzero()[0]]
print(df)



# get parameters for GridSearch
scorer = c.f1(average="micro") # use F1 score with micro averaging
estimator = c.knn() # get kNN estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    "n_neighbors":range(3,50)
    ,"algorithm":[
            "auto"
            #,"ball_tree"
            #,"kd_tree"
            #,"brute"
    ]
    ,"weights":[
            #"uniform"
            "distance"
    ]
    ,"p":[
            #1
            2
            #,3
    ]
    ,"metric":[
            "euclidean"
            #,"manhattan"
            #,"chebyshev"
            #,"minkowski"
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

# Best score is 0.3237997957099081 with params {'algorithm': 'auto', 'metric': 'euclidean', 'n_neighbors': 9, 'p': 2, 'weights': 'distance'}.

# save all results in csv
c.gridSearchResults2CSV(gs,parameters,"results_kNN.csv")