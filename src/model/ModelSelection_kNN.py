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
c = ct.Classifier(df,"productivity_binned_binary")

### drop single columns not needed for Classification
c.dropColumns([
        "original_title"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
        ,"quarter"
        
        ,"adult"
        #,"belongs_to_collection"
        #,"budget"
        ,"runtime"
        #,"year"
])

### scale something if needed
#c.scale([
#        "budget"
#])

### drop columns by prefix if needed
c.dropColumnByPrefix("actor")
#c.dropColumnByPrefix("director")
#c.dropColumnByPrefix("company")
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
c.dropColumnByPrefix("quarter_")

df = c.data.loc[19898]
df = df.iloc[df.nonzero()[0]]
print(df)
print(c.data.columns)

print(len(c.data.columns))
c.thresholdByColumn(3,"company")
c.thresholdByColumn(8,"actor")
c.thresholdByColumn(3,"director")
print(len(c.data.columns))

# lets print all non-zero columns of a movie to doublecheck
df = c.data.loc[19898]
df = df.iloc[df.nonzero()[0]]
print(df)
print(c.data.columns)

# get information about the data
c.balanceInfo()

# get parameters for GridSearch
scorer = c.f1(average="macro") # use F1 score with micro averaging
estimator = c.knn() # get kNN estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    "n_neighbors":[16]
    ,"algorithm":[
            "auto"
            #,"ball_tree"
            #,"kd_tree"
            #,"brute"
    ]
    ,"weights":[
            "uniform"
            #"distance"
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



# calculate cross validation: try samplings
estimator.set_params(
        n_neighbors=16
        ,algorithm="auto"
        ,weights="uniform"
        ,p=2
        ,metric="euclidean"
)
print(c.cross_validate(cv,estimator,sample=""))
#print(c.cross_validate(cv,estimator,sample="down"))
#print(c.cross_validate(cv,estimator,sample="up"))


"""
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
c.gridSearchResults2CSV(gs,parameters,"results_kNN.csv")
"""

"""
--------------------------- GRID SEARCH BEST SCORE ---------------------------
 Best score is 0.603718127805059 with params {'algorithm': 'auto', 'metric': 'euclidean', 'n_neighbors': 16, 'p': 2, 'weights': 'uniform'}.
 ------------------------------------------------------------------------------



CROSS VALIDATION PARAMETERS AND VALUE

c.dropColumns([
        "original_title"
        ,"productivity_binned_multi"
        ,"quarter"
        ,"adult"
        ,"runtime"
])

c.dropColumnByPrefix("actor")
c.dropColumnByPrefix("country")
c.dropColumnByPrefix("genre")
c.dropColumnByPrefix("quarter_")

c.thresholdByColumn(3,"company")
c.thresholdByColumn(8,"actor")
c.thresholdByColumn(3,"director")

estimator.set_params(
        n_neighbors=16
        ,algorithm="auto"
        ,weights="uniform"
        ,p=2
        ,metric="euclidean"
)
unsampled: {'f1': 0.6140887381748299, 'no': 1067, 'yes': 2846}
up-sampled: {'f1': 0.47963743779714313, 'no': 2648, 'yes': 1265}
down-sampled: {'f1': 0.39577490421400813, 'no': 3128, 'yes': 785}
"""