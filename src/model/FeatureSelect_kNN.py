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

label_column = "productivity_binned_binary"

# Build Classifier object with DataFrame and column name of truth values
c = ct.Classifier(df,label_column)

### drop single columns not needed for Classification
c.dropColumns([
        "original_title"
        #,"adult"
        #,"belongs_to_collection"
        #,"budget"
        #,"runtime"
        #,"year"
        ,"quarter"
        ,"productivity_binned_multi"
        #,"productivity_binned_binary"
])

### scale something if needed
#c.scale([
#        "budget"
#])

### drop columns by prefix if needed
#c.dropColumnByPrefix("actor")
#c.dropColumnByPrefix("director")
#c.dropColumnByPrefix("company")
#c.dropColumnByPrefix("country")
#c.dropColumnByPrefix("genre")
#c.dropColumnByPrefix("quarter_")

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
    "n_neighbors":range(5,6)
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

features = [
            "adult",
            "belongs_to_collection",
            "budget",
            "runtime",
            "year",
            "actor_",
            "director_",
            "company_",
            "country_",
            "genre_",
            "quarter_"
]

# compute FeatureSelect
gs = c.featureselect_greedy(
        features
        ,parameters
        ,scorer
        ,estimator
        ,cv
        ,label_column
)

#print(gs)




"""
    CURRENT BEST STATS
    -------------
    "n_neighbors":range(5,6)
    ,"algorithm":[
            "auto"
    ]
    ,"weights":[
            "distance"
    ]
    ,"p":[
            2
    ]
    ,"metric":[
            "euclidean"
    -------------
    CURRENT: 0.5793221353439061, MAX: 0.5766128624882128, FEATURE: country_
    DROPPED: ['genre_', 'budget', 'quarter_', 'adult', 'actor_', 'director_']
"""