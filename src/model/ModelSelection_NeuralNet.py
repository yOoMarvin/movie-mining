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
        "original_title",
])

c.scale([
        "budget"
])

#c.dropColumnByPrefix("actor")
#c.dropColumnByPrefix("director")
#c.dropColumnByPrefix("company")
#c.dropColumnByPrefix("country")
#c.dropColumnByPrefix("genre")
#c.dropColumnByPrefix("quarter")

df = c.data.loc[19898]
df = df.iloc[df.nonzero()[0]]
print(df)



# get parameters for GridSearch
scorer = c.f1(average="micro") # use F1 score with micro averaging
estimator = c.neuralnet() # get estimator
cv = c.fold(
        k=10
        ,random_state=42
) # KStratifiedFold with random_state = 42
# parameters to iterate in GridSearch
parameters = {
    "solver":[
            "lbfgs"
            ,"sgd"
            ,"adam"
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

# Best score is 0.35137895812053116 with params {'solver': 'sgd'}.
# Best score is 0.394535240040858 with params {'solver': 'sgd'}. - with all columns
# Best score is 0.39096016343207357 with params {'solver': 'sgd'}. - all columns + budget

# save all results in csv
c.gridSearchResults2CSV(gs,parameters,"results_NeuralNet.csv")